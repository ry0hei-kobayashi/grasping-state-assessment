# Grasping-state-assessment

本リポジトリは論文 **"Grasp Status Assessment of Deformable Objects Using Visual‑Tactile Fusion"** の再現実験用コードです。

この README では `train.py` をベースに、**入力データの形・ネットワーク構成（どこで融合しているか）・学習の流れ・実行コマンド** をコードに即して整理します。

---

## 1. 学習スクリプト全体像（`train.py`）

`train.py` の学習ループは基本的に以下です。

- **データ**: `xela_dataloader.MyDataset` が `(x_visual, x_tactile, y)` を返す
- **モデル**: `--model_arch` で分岐（`models/models.py` 内クラス）
- **損失**: `torch.nn.functional.cross_entropy(outputs, targets)`
- **最適化**: `torch.optim.SGD(model.parameters(), lr=opt.lr)`
- **評価指標**: `sklearn.metrics.accuracy_score`
- **保存**: `opt.checkpoint` 配下に `checkpoint.pth.tar` / `model_best.pth.tar`

学習・評価はそれぞれ `train()` / `valid()` で実施され、各 epoch の `train_loss/train_acc` と `valid_loss/valid_acc` をログに残します。

---

## 2. 入力データとテンソル形状（重要）

### 2.1 データセット構造（`xela_dataloader.py`）

`MyDataset(dataroot, visual_seq_length, tactile_seq_length, ..., flag)` は `dataroot` 以下の各カテゴリフォルダからサンプルを生成します。

例（実際に repo に存在する構造）:

- `graspingdata/appbox/<case>/visual/*.jpg`
- `graspingdata/appbox/<case>/tactile/*.jpg`
- ...（`baisui`, `bingho`, `cesbon`, ...）

`case` は `width_force_label` のようにアンダースコア区切りで、最後の `label` が教師ラベルとして使われます。

### 2.2 DataLoader が返す shape

`__getitem__` は以下を返します（概念的表現）:

- **visual**: `x_visual` 形状 `(3, T_v, H, W)`（RGB × visual_seq_length）
- **tactile**: `x_tactile` 形状 `(3, T_t, 4, 4)`（RGB × tactile_seq_length、`train.py` 側で 4x4 にリサイズ）
- **label**: `targets` 形状 `()`（`torch.long`）

DataLoader によりバッチ化されると、

- `x_visual`: `(N, 3, T_v, H, W)`
- `x_tactile`: `(N, 3, T_t, 4, 4)`

になります。

### 2.3 `C3D` に入力するための整形（train.py の実装）

`C3D` 系（3D-CNN）は一般に `Conv3d` 入力 `shape=(N, C, D, H, W)` を前提にします。

本 repo の `C3D` 実装では **RGB(3) を「Depth(D)」側へ畳み込み**、以下のように変換してから `model()` に渡します（`train.py` に実装済み）。

- `(N, 3, T, H, W)` → `(N, 1, 3*T, H, W)`

これにより `models/models.py` 内 `CNN3D` / `CNN3D1` が想定する `in_channels=1` と整合します。

---

## 3. ネットワーク構成（`models/models.py`）

`train.py` は `--model_arch` に応じてモデルを生成します。実装には複数の融合方式があります。

ここでは **「どの段で Visual と Tactile を融合しているか」** という観点で整理します。

### 3.1 C3D（推奨: train.py と整合して学習を回しやすい）

`--model_arch C3D`

- **Visual branch**: `CNN3D`（3D Conv → FC）
- **Tactile branch**: `CNN3D1`（3D Conv → FC）
- **Fusion**: 2つの埋め込みを `concat` → `fc1` → `fc2`（分類）

特徴:

- 入力が「画像列」でもそのまま学習できる（特徴抽出器を別途用意しない）
- 視覚と触覚を **late fusion（最終段近く）** で統合

### 3.2 EarlyFusion / LateFusion / ModalFN / MARN / VTFSA_LSTM（研究用の融合バリエーション）

これらは大きく分けて以下の設計です。

- **EarlyFusion系**: 早い段で `concat(x_visual, x_tactile)` してから LSTM 等に入れる
- **LateFusion系**: 各モダリティを別々に時系列エンコードしてから最後に結合
- **ModalFN / MARN**: 追加の LSTM ブロックや Attention/Hybrid state を介して統合
- **VTFSA_LSTM**: feature map 同士を空間的に融合し Self-Attention をかけ、LSTM で時系列統合

注意:

- これらは **入力が「すでに特徴ベクトル列」**（例: `(N,T,2048)`）である想定が混ざっており、`train.py` の現状の「画像列」入力とはそのままだと整合しない場合があります。
- まずは **C3D を動く基準系**として学習が回ることを確認し、その後必要に応じて特徴抽出器（ResNet 等）を追加して他モデルを試すのが安全です。

---

## 4. 学習を回す手順

### 4.1 前提: データ配置

デフォルトでは `--dataroot ./graspingdata` を参照します（`options.py`）。

リポジトリ直下に `graspingdata/` があり、カテゴリフォルダ（`appbox/` など）が入っている必要があります。

### 4.2 代表コマンド（C3Dで学習）

例: GPU 0 を使って学習し、`checkpoint_sgd/exp_c3d/` に結果を保存する:

```bash
python3 train.py \
  --model_arch C3D \
  --dataroot ./graspingdata \
  --checkpoint checkpoint_sgd \
  --name exp_c3d \
  --batchSize 8 \
  --lr 1e-4 \
  --epochs 200 \
  --gpu_ids 0
```

### 4.3 評価のみ（保存済みモデルを使う）

```bash
python3 train.py \
  --evaluate \
  --model_arch C3D \
  --dataroot ./graspingdata \
  --resume checkpoint_sgd/exp_c3d/model_best.pth.tar \
  --gpu_ids 0
```

---

## 8. 依存関係（最低限）

実行環境には少なくとも以下が必要です。

- `python3`
- `torch`, `torchvision`
- `Pillow`
- `numpy`
- `matplotlib`（学習後の `plt.show()` を使う場合）

※ `xela_dataloader.py` はラベルエンコードに `scikit-learn` を使っています（`LabelEncoder`, `OneHotEncoder`）。

---

## 5. 出力物

学習が進むと `--checkpoint <dir> --name <exp>` で指定した出力先（例: `checkpoint_sgd/exp_c3d/`）に以下が生成されます。

- **`opt.txt`**: 実行時オプションのダンプ
- **`log.txt`**: 学習ログ（LR / Train Loss / Valid Loss / Valid Acc）
- **`checkpoint.pth.tar`**: 最新 checkpoint
- **`model_best.pth.tar`**: validation accuracy が最大の checkpoint

また、学習完了後に `XELA_results/` に以下が保存されます:

- `train_acc_*.npy`, `train_loss_*.npy`, `test_acc_*.npy`, `test_loss_*.npy`

---

## 6. よくあるハマりどころ（コードベース由来）

- **`models/models.py` の import**: `LSTHM` や `resnet` は repo 内実装に依存するため、相対 import で解決する必要があります（本 repo では修正済み）。
- **テンソル形状の不一致**: `C3D` は `Conv3d` 前提の 5Dテンソル入力のため、`(N,3,T,H,W)` を `(N,1,3*T,H,W)` に変換しないと shape mismatch になります（`train.py` で対応済み）。
- **クラス数**: データの `label` が `0/1/2` を含む場合、`num_classes` を 3 に揃える必要があります（モデル側の設定とラベル定義が一致しているか要確認）。

---

## 7. 参考（外部配布ファイル）

不足ファイル（npy形式データ等）と対応 dataloader は Google Drive で共有されています:

`https://drive.google.com/drive/folders/1ialG2_z73bw0xYB7KpoNe-kmfED-G1p3?usp=sharing`

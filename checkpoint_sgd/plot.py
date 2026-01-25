import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 設定 =====
LOG_PATH = "200_epoch_log_best.tsv"
out_dir = Path("figs")
out_dir.mkdir(exist_ok=True)

# ===== 読み込み（TSV）=====
df = pd.read_csv(LOG_PATH, sep="\t")

# 余計な空列（末尾タブで Unnamed 列ができることがある）を落とす
df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

# 列名の前後スペース除去
df.columns = df.columns.str.strip()

# Epoch列を作る（ログがepoch順に並んでいる前提）
df["Epoch"] = range(1, len(df) + 1)

# 欠損/文字列混入に備えて数値化（失敗はNaN）
for c in df.columns:
    if c != "Epoch":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ===== 1) Loss =====
plt.figure()
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig(out_dir / "train_loss.png", dpi=300, bbox_inches="tight")
plt.close()

# ===== 1) Loss =====
plt.figure()
plt.plot(df["Epoch"], df["Test Loss"], label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig(out_dir / "test_loss.png", dpi=300, bbox_inches="tight")
plt.close()

# ===== 2) Accuracy =====
plt.figure()
plt.plot(df["Epoch"], df["Train Acc"], label="Train Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(out_dir / "train_acc.png", dpi=300, bbox_inches="tight")
plt.close()

# ===== 2) Accuracy =====
plt.figure()
plt.plot(df["Epoch"], df["Test Acc"], label="Test Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig(out_dir / "test_acc.png", dpi=300, bbox_inches="tight")
plt.close()

# ===== 3) VRAM 使用量 =====
# どれか1つでも列があれば描画（環境で列名が微妙に違うケース対策）
vram_cols = [
    c
    for c in ["VRAM_used(MB)", "VRAM_peak_alloc(MB)", "VRAM_peak_reserved(MB)"]
    if c in df.columns
]
if vram_cols:
    plt.figure()
    for c in vram_cols:
        plt.plot(df["Epoch"], df[c], label=c)
    plt.xlabel("Epoch")
    plt.ylabel("MB")
    plt.title("VRAM (MB) vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(out_dir / "vram.png", dpi=300, bbox_inches="tight")
    plt.close()

# ===== 4) Test Best Acc の確認（任意）=====
if "Best Acc (Test)." in df.columns:
    best = df["Best Acc (Test)."].max()
    best_epoch = df.loc[df["Best Acc (Test)."] == best, "Epoch"].iloc[0]
    print(f"Best Acc (Test). = {best:.6f} @ Epoch {best_epoch}")

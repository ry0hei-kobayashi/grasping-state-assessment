# CUI's Grasping State Assesment

## 仮装環境

- Ubuntu22.04
- Cuda 11.7

## How to build

```bash
cd Grasping-state-assessment/env/
apptainer build --fakeroot --sandbox sandbox_gsa_cu117 grasp_state_assesment_cu117.def
```

## How to run

```bash
apptainer shell --nv sandbox_gsa_cu117
source /entrypoint.sh
```

## Grasping State Assesmentの実行

- リポジトリをクローンする

```bash
git clone https://github.com/yuzoo0226/Grasping-state-assessment.git
cd Grasping-state-assessment
git checkout yano/hotfix
```

## How to train

```bash
python3 train.py --model_arch C3D --dataroot ./graspingdata --checkpoint checkpoint_sgd --name exp_c3d --batchSize 8 --lr 1e-4 --epochs 200 --gpu_ids 0
```

## How to test

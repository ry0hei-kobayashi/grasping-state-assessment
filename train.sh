python3 train.py \
  --model_arch C3D \
  --dataroot ./graspingdata \
  --checkpoint checkpoint_sgd \
  --name exp_c3d \
  --batchSize 8 \
  --lr 1e-4 \
  --epochs 200 \
  --gpu_ids 0 


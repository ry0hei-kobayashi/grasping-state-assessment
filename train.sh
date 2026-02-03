python3 train.py \
  --model_arch C3D \
  --dataroot ./graspingdata \
  --name c3d_200_3class \
  --batchSize 8 \
  --lr 1e-4 \
  --epochs 200 \
  --gpu_ids 0 

#python3 train.py \
#  --model_arch C3D \
#  --dataroot ./graspingdata \
#  --checkpoint checkpoint_sgd \
#  --name c3d_200 \
#  --batchSize 4 \
#  --lr 1e-4 \
#  --epochs 200 \
#  --gpu_ids 0 


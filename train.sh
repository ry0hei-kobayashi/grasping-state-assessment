python3 train.py \
  --model_arch C3D \
  --dataroot ./graspingdata \
  --name c3d_all \
  --batchSize 8 \
  --lr 1e-7 \
  --epochs 200 \
  --gpu_ids 1 \
  --train_classes appbox,bingho,cokele,jianjo,nestle,nongfu,pacup2,zhijin,baisui,cesbon,haitun,meinad \
  --test_classes pacup1,nongf1,songsu

#appbox  bingho  cokele  jianjo  nestle  nongfu  pacup2  zhijin
#baisui  cesbon  haitun  meinad  nongf1  pacup1  songsu


#python3 train.py \
#  --model_arch C3D \
#  --dataroot ./graspingdata \
#  --checkpoint checkpoint_sgd \
#  --name c3d_200 \
#  --batchSize 4 \
#  --lr 1e-4 \
#  --epochs 200 \
#  --gpu_ids 0 


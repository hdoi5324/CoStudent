#!/bin/bash

lrs=(0.01 0.001) # 0.02 for FR batch 16 0.01 for FCOS batch 16 check lr for DETR 0.0001??
ims_per_gpu=2
SEEDS=(1234575 2234575 3234575) 

data_vol="few" # few gtfew
image_type="og" # og cropped
species="urchin" # redcup urchin ob
loss_type="giou"
for lr in "${lrs[@]}"
do 
  for SEED in "${SEEDS[@]}"
  do
    python tools/train_net.py --dir configs/CoStudent_fcos.res50.score.05.teacher_score.06.${species}.${data_vol}.${image_type} \
    SOLVER.OPTIMIZER.BASE_LR ${lr} SOLVER.IMS_PER_DEVICE ${ims_per_gpu} MODEL.FCOS.IOU_LOSS_TYPE ${loss_type} SEED ${SEED} \
    OUTPUT_DIR outputs/${species}_${data_vol}_${image_type}_${loss_type}_${lr}_${SEED}
  done
done
#!/bin/bash

lrs=(0.005 0.001) # 0.02 for FR batch 16 0.01 for FCOS batch 16 check lr for DETR 0.0001??
ims_per_gpu=16
SEEDS=(1234575 2234575 3234575) #

species="anemone" # redcup urchin ob
lr=0.01
for SEED in "${SEEDS[@]}"
do
  for lr in "${lrs[@]}"
  do
    bash scripts/train_with_dataset.sh --train-images /media/data/phd_data/loose_sam3/20260421-173150_redcup-sparse-coco/images \
    --train-json /media/data/phd_data/loose_sam3/20260421-173150_redcup-sparse-coco/sparse_point_coco.json \
    --test-images /media/data/phd_data/squidle_coco/squidle_redcup_test/test2023 \
    --test-json /media/data/phd_data/squidle_coco/squidle_redcup_test/annotations/instances_test.json \
    -- OUTPUT_DIR outputs/redcup_sam3_og_giou_${lr}_${SEED} SOLVER.OPTIMIZER.BASE_LR ${lr} SOLVER.IMS_PER_DEVICE ${ims_per_gpu} SEED ${SEED}
  done
done

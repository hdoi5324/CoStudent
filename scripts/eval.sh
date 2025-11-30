#!/bin/bash

species="anemone"
iter="0001500"
for image in "cropped" "og"
do
  for seed in 1234575 2234575 3234575
  do
    python tools/test_net.py --dir configs/CoStudent_fcos.res50.${species}.few.${image} MODEL.WEIGHTS outputs/${species}_few_${image}_giou_0.01_${seed}/teacher_model_iter_${iter}.pth OUTPUT_DIR outputs/${species}_few_${image}_giou_0.01_${seed}/
    #python tools/visualize_json_results.py --dir configs/CoStudent_fcos.res50.${species}.few.${image} --input ./outputs/${species}_few_${image}_giou_0.01_${seed}/inference/coco_instances_results.json --output ./outputs/${species}_few_${image}_giou_0.01_${seed}/inference --conf-threshold 0.5 --dataset coco_squidle_anemone_full_test
  done
 done 
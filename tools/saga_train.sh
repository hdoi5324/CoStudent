#!/bin/bash

SEEDS=(1234575 3234575 2234575)
lr=0.0125
for SEED in "${SEEDS[@]}"
do
  #sbatch tools/saga_slurm_train_net.sh configs/CoStudent_fcos.res50.score.05.teacher_score.06.urchin.few.og "SOLVER.OPTIMIZER.BASE_LR ${lr} SEED ${SEED} OUTPUT_DIR \'outputs/loose/urchin_og_few_costudent_nta_${SEED}_${lr}/\'"
  #sbatch tools/saga_slurm_train_net.sh configs/CoStudent_fcos.res50.score.05.teacher_score.06.redcup.few.og "SOLVER.OPTIMIZER.BASE_LR ${lr} SEED ${SEED} OUTPUT_DIR \'outputs/loose/redcup_og_few_costudent_nta_${SEED}_${lr}/\'"

  #sbatch tools/saga_slurm_train_net.sh configs/CoStudent_fcos.res50.score.05.teacher_score.06.urchin.gtfew.og "SOLVER.OPTIMIZER.BASE_LR ${lr} SEED ${SEED} OUTPUT_DIR \'outputs/loose/urchin_og_gtfew_costudent_nta_${SEED}_${lr}/\'"
  #sbatch tools/saga_slurm_train_net.sh configs/CoStudent_fcos.res50.score.05.teacher_score.06.redcup.gtfew.og "SOLVER.OPTIMIZER.BASE_LR ${lr} SEED ${SEED} OUTPUT_DIR \'outputs/loose/redcup_og_gtfew_costudent_nta_${SEED}_${lr}/\'"

#  sbatch tools/saga_slurm_train_net.sh configs/CoStudent_fcos.res50.score.05.teacher_score.06.urchin.few.cropped "SOLVER.OPTIMIZER.BASE_LR ${lr} SEED ${SEED} OUTPUT_DIR \'outputs/loose/urchin_cropped_few_costudent_nta_${SEED}_${lr}/\'"
  sbatch tools/saga_slurm_train_net.sh configs/CoStudent_fcos.res50.score.05.teacher_score.06.redcup.few.cropped "SOLVER.OPTIMIZER.BASE_LR ${lr} SEED ${SEED} OUTPUT_DIR \'outputs/loose/redcup_cropped_few_costudent_nta_${SEED}_${lr}/\'"

done
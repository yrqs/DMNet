#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2
PORT=${PORT:-24500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=1,3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/ga_retina_dml7_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_base1_r1_lr00025x2x2_10_14_16_ind3_1'
#    --work_dir 'work_dirs/ga_retina_dml7_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_aug2_standard1_2shot_r25_lr0001x2x1_warm500_14_18_20_ind1_1_1'
#    --work_dir 'work_dirs/ga_retina_dml3_fpn_emb256_64_alpha015_le10_CE_nratio10_coco_aug_standard_10shot_r10_lr0001x2x2_14_18_20_ind2_1_1/'
#    --work_dir 'work_dirs/ga_retina_dml3_fpn_emb256_64_alpha015_le10_CE_nratio3_coco_base_r1_lr00025x2x2_20_24_26_ind3_1'
#    --work_dir 'work_dirs/ga_retina_dml3_fpn_emb256_64_alpha015_le10_CE_nratio3_coco_base_r1_lr00025x2x2_12_16_18_ind1_1'
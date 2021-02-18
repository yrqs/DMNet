#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=configs/few_shot/voc/retina_dml2_r101_fpn_standard1_shot/retina_dml2_r101_fpn_standard1_1shot.py
GPUS=1
PORT=${PORT:-87500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/retina_dml2_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_aug_standard1_1shot_lr0001x2x1_warm1000_12_16_18_ind1_1_1'

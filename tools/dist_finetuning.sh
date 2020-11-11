#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2
PORT=${PORT:-39500}

#    --work_dir './work_dirs/ga_dml_rpn_x101_uw20_embsize128_alpha015_lr0005-7-10-13_CE_ind1_beta004_max'
CUDA_VISIBLE_DEVICES=3 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/finetune_dml.py $CONFIG --launcher pytorch ${@:3} \
    --work_dir './work_dirs/finetuning/test3'

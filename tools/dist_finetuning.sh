#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2
PORT=${PORT:-27500}

OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/finetune.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --shot 10 \
    --work_dir 'work_dirs/test'
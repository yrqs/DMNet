#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2
PORT=${PORT:-40500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0,1 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/ga_retina_dml2_normFalse_voc_base2'

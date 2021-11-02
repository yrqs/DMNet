#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2
PORT=${PORT:-60500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=4,5 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_norm_voc_standard2_3shot'

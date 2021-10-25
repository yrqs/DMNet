#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-79500}

CUDA_VISIBLE_DEVICES=0 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch ${@:4} \
    --out test_out.pkl \
    --eval mAP
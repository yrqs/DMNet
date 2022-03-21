#!/usr/bin/env bash

DATASET='coco'
MODEL_NAME=ga_retina_dml4_coco
PARAMETER=wo_norm

CONFIG_PATH='configs/few_shot/'$DATASET'/'$MODEL_NAME'/'$PARAMETER'/'
WORK_DIR_BASE='work_dirs/'$MODEL_NAME'/'$PARAMETER'/'
PORT=${PORT:-42500}
PYTHON=${PYTHON:-"python"}

CONFIG=$CONFIG_PATH'base.py'
GPUS=4
OMP_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES=4,5,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'base'

CONFIG_BASE=$CONFIG_PATH'finetune'

GPU_ID=4,5,6,7
GPUS=4
CONFIG=${CONFIG_BASE}_10shot.py

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'10shot' \

CONFIG=${CONFIG_BASE}_30shot.py
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'30shot'
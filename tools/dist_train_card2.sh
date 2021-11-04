#!/usr/bin/env bash

CONFIG_BASE=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre01_r101_fpn_standard2/finetune
WORK_DIR_BASE='work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre01_r101_lr0001x2x1_voc_standard2'


GPU_ID=4,5
GPUS=2
PYTHON=${PYTHON:-"python"}
CONFIG=${CONFIG_BASE}_1shot.py
PORT=${PORT:-47500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'_1shot'

PYTHON=${PYTHON:-"python"}
CONFIG=${CONFIG_BASE}_2shot.py
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'_2shot'

PYTHON=${PYTHON:-"python"}
CONFIG=${CONFIG_BASE}_3shot.py
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'_3shot'

PYTHON=${PYTHON:-"python"}
CONFIG=${CONFIG_BASE}_5shot.py
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'_5shot'

PYTHON=${PYTHON:-"python"}
CONFIG=${CONFIG_BASE}_10shot.py
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'_10shot'
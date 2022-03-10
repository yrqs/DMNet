#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2
PORT=${PORT:-27500}
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'test'
#    --work_dir 'work_dirs/ga_retina_dml_force3_voc_base1'
#
#
#CONFIG_BASE=configs/few_shot/voc/ga_retina_dml2_r101_fpn_standard2/finetune
#WORK_DIR_BASE='work_dirs/ga_retina_dml2_r101_lr0001x2x1_voc_standard2'
#
#PYTHON=${PYTHON:-"python"}
#CONFIG=${CONFIG_BASE}_1shot.py
#GPUS=1
#PORT=${PORT:-87500}
#OMP_NUM_THREADS=1 \
#CUDA_VISIBLE_DEVICES=7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#    --validate \
#    --work_dir $WORK_DIR_BASE'_1shot'
#
#PYTHON=${PYTHON:-"python"}
#CONFIG=${CONFIG_BASE}_2shot.py
#GPUS=1
#PORT=${PORT:-87500}
#OMP_NUM_THREADS=1 \
#CUDA_VISIBLE_DEVICES=7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#    --validate \
#    --work_dir $WORK_DIR_BASE'_2shot'
#
#PYTHON=${PYTHON:-"python"}
#CONFIG=${CONFIG_BASE}_3shot.py
#GPUS=1
#PORT=${PORT:-87500}
#OMP_NUM_THREADS=1 \
#CUDA_VISIBLE_DEVICES=7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#    --validate \
#    --work_dir $WORK_DIR_BASE'_3shot'
#
#PYTHON=${PYTHON:-"python"}
#CONFIG=${CONFIG_BASE}_5shot.py
#GPUS=1
#PORT=${PORT:-87500}
#OMP_NUM_THREADS=1 \
#CUDA_VISIBLE_DEVICES=7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#    --validate \
#    --work_dir $WORK_DIR_BASE'_5shot'
#
#PYTHON=${PYTHON:-"python"}
#CONFIG=${CONFIG_BASE}_10shot.py
#GPUS=1
#PORT=${PORT:-87500}
#OMP_NUM_THREADS=1 \
#CUDA_VISIBLE_DEVICES=7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#    --validate \
#    --work_dir $WORK_DIR_BASE'_10shot'

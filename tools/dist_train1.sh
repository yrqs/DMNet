#!/usr/bin/env bash

#PYTHON=${PYTHON:-"python"}
#CONFIG=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_1333_standard2/base.py
#GPUS=4
#PORT=${PORT:-77500}
#OMP_NUM_THREADS=2 \
#CUDA_VISIBLE_DEVICES=2,3,4,5 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#    --validate \
#    --work_dir 'work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_1333_r101_voc_base2'

CONFIG_BASE=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_fpn_standard2/finetune
WORK_DIR_BASE='work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_voc_standard2_1108_ind2'

GPU_ID=0,1,2,3,4,5,6,7
GPUS=8
PORT=${PORT:-37500}

#PYTHON=${PYTHON:-"python"}
#CONFIG=${CONFIG_BASE}_1shot.py
#OMP_NUM_THREADS=1 \
#CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#    --validate \
#    --work_dir $WORK_DIR_BASE'_1shot'
#
#PYTHON=${PYTHON:-"python"}
#CONFIG=${CONFIG_BASE}_2shot.py
#OMP_NUM_THREADS=1 \
#CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#    --validate \
#    --work_dir $WORK_DIR_BASE'_2shot'
#
#PYTHON=${PYTHON:-"python"}
#CONFIG=${CONFIG_BASE}_3shot.py
#OMP_NUM_THREADS=1 \
#CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#    --validate \
#    --work_dir $WORK_DIR_BASE'_3shot'
#
#PYTHON=${PYTHON:-"python"}
#CONFIG=${CONFIG_BASE}_5shot.py
#OMP_NUM_THREADS=1 \
#CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
#    --validate \
#    --work_dir $WORK_DIR_BASE'_5shot'

PYTHON=${PYTHON:-"python"}
CONFIG=${CONFIG_BASE}_10shot.py
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'_10shot'
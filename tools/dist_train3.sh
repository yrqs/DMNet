#!/usr/bin/env bash


PYTHON=${PYTHON:-"python"}
CONFIG=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre03_r101_fpn_standard2/finetune_3shot.py
GPUS=2
PORT=${PORT:-47500}
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=4,5 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/ga_retina_nscope20_nalpha01_nthre03_voc_standard2_3shot'
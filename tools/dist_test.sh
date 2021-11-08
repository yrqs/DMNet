#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

#CONFIG=$1
#CHECKPOINT=$2
#GPUS=$3

CONFIG=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_fpn_standard2/base.py
CHECKPOINT=work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_voc_standard1_10shot/epoch_16.pth
GPUS=1
PORT=${PORT:-79500}

CUDA_VISIBLE_DEVICES=5 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch ${@:4} \
    --out test_out.pkl \
    --eval mAP

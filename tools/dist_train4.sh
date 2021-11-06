#!/usr/bin/env bash


PYTHON=${PYTHON:-"python"}
CONFIG=configs/few_shot/coco/ga_retina_dmlneg3_nscope20_nalpha01_nthre02/base.py
GPUS=4
PORT=${PORT:-57500}
OMP_NUM_THREADS=4 \
CUDA_VISIBLE_DEVICES=0,1,4,5 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_coco_base'
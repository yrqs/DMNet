#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
#CHECKPOINT=work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_voc_standard1_10shot/epoch_16.pth

#CONFIG=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_fpn_standard1/finetune_1shot.py
#CONFIG=configs/few_shot/voc/retinanet_r101_fpn_standard1.py
#CHECKPOINT=work_dirs/retina_s2_fpn_voc_aug_standard1_10shot_r10_lr00025x2x1_warm1000_12_16_18_ind1_1_1/epoch_18.pth
#CHECKPOINT=work_dirs/retina_s2_fpn_voc_aug_standard1_1shot_r30_lr00025x2x1_warm1000_12_16_18_ind1_1_1/epoch_16.pth
#GPUS=1
PORT=${PORT:-79500}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch ${@:4} \
    --out test.pkl \
    --eval mAP
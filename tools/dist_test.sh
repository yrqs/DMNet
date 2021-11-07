#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

#CONFIG=$1
#CHECKPOINT=$2
#GPUS=$3

CONFIG=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_fpn_standard1/finetune_1shot.py
CHECKPOINT=work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_voc_standard3_1shot/epoch_14.pth
GPUS=4
PORT=${PORT:-79500}

CUDA_VISIBLE_DEVICES=2,3,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch ${@:4} \
    --out test_out.pkl \
    --eval mAP

CONFIG=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_fpn_standard1/finetune_1shot.py
CHECKPOINT=work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_voc_standard3_2shot/epoch_14.pth
GPUS=4
PORT=${PORT:-79500}

CUDA_VISIBLE_DEVICES=2,3,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch ${@:4} \
    --out test_out.pkl \
    --eval mAP

CONFIG=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_fpn_standard1/finetune_1shot.py
CHECKPOINT=work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_voc_standard3_3shot/epoch_10.pth
GPUS=4
PORT=${PORT:-79500}

CUDA_VISIBLE_DEVICES=2,3,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch ${@:4} \
    --out test_out.pkl \
    --eval mAP

CONFIG=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_fpn_standard1/finetune_1shot.py
CHECKPOINT=work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_voc_standard3_5shot/epoch_6.pth
GPUS=4
PORT=${PORT:-79500}

CUDA_VISIBLE_DEVICES=2,3,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch ${@:4} \
    --out test_out.pkl \
    --eval mAP

CONFIG=configs/few_shot/voc/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_fpn_standard1/finetune_1shot.py
CHECKPOINT=work_dirs/ga_retina_dmlneg3_nscope20_nalpha01_nthre02_r101_voc_standard3_10shot/epoch_14.pth
GPUS=4
PORT=${PORT:-79500}

CUDA_VISIBLE_DEVICES=2,3,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT \
    --launcher pytorch ${@:4} \
    --out test_out.pkl \
    --eval mAP
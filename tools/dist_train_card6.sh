#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=configs/few_shot/voc/ga_retina_dml3_r101_fpn_standard1_alpha_shot/ga_retina_dml3_r101_fpn_standard1_007_1shot.py
GPUS=1
PORT=${PORT:-86500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=6 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/ga_retina_dml3_s2_fpn_emb256_128_alpha007_le10_CE_nratio3_voc_aug_standard1_1shot_lr00025x2x1_warm1000_12_16_18_ind1_1_1'

PYTHON=${PYTHON:-"python"}
CONFIG=configs/few_shot/voc/ga_retina_dml3_r101_fpn_standard1_alpha_shot/ga_retina_dml3_r101_fpn_standard1_007_2shot.py
GPUS=1
PORT=${PORT:-86500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=6 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/ga_retina_dml3_s2_fpn_emb256_128_alpha007_le10_CE_nratio3_voc_aug_standard1_2shot_lr00025x2x1_warm1000_12_16_18_ind1_1_1'

PYTHON=${PYTHON:-"python"}
CONFIG=configs/few_shot/voc/ga_retina_dml3_r101_fpn_standard1_alpha_shot/ga_retina_dml3_r101_fpn_standard1_007_3shot.py
GPUS=1
PORT=${PORT:-86500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=6 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/ga_retina_dml3_s2_fpn_emb256_128_alpha007_le10_CE_nratio3_voc_aug_standard1_3shot_lr00025x2x1_warm1000_12_16_18_ind1_1_1'

PYTHON=${PYTHON:-"python"}
CONFIG=configs/few_shot/voc/ga_retina_dml3_r101_fpn_standard1_alpha_shot/ga_retina_dml3_r101_fpn_standard1_007_5shot.py
GPUS=1
PORT=${PORT:-86500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=6 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/ga_retina_dml3_s2_fpn_emb256_128_alpha007_le10_CE_nratio3_voc_aug_standard1_5shot_lr00025x2x1_warm1000_12_16_18_ind1_1_1'

PYTHON=${PYTHON:-"python"}
CONFIG=configs/few_shot/voc/ga_retina_dml3_r101_fpn_standard1_alpha_shot/ga_retina_dml3_r101_fpn_standard1_007_10shot.py
GPUS=1
PORT=${PORT:-86500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=6 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/ga_retina_dml3_s2_fpn_emb256_128_alpha007_le10_CE_nratio3_voc_aug_standard1_10shot_lr00025x2x1_warm1000_12_16_18_ind1_1_1'

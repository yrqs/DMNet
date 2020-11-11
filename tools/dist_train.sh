#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2
PORT=${PORT:-47500}

#    --work_dir './work_dirs/ga_dml_rpn_x101_uw20_embsize128_alpha015_lr0005-7-10-13_CE_ind1_beta004_max'
#    --work_dir './work_dirs/ga_retina_x101_voc_embsize128_alpha015_lr0005-7-11-13_CE_ind1_beta004_tloss4_10s'
#    --work_dir './work_dirs/ga_dml_rpn_x101_voc_embsize128_alpha015_lr000125-15-24-30_CE_ind1_beta004_tloss4_10s'
#    --work_dir './work_dirs/ga_retina_x101_voc_ind1_beta004_lr00025-10-20-30_10s'
#    --work_dir './work_dirs/ga_dml_rpn_x101_voc_embsize512_256_alpha015_lr000125-36-50-60_CE_ind1_beta004_tloss4_m3_t10s_aug_r5'
#    --work_dir './work_dirs/ga_dml_rpn2_x101_voc_embsize1024_256_alpha015_lr000125-34-40-44_CE_ind1_beta004_tloss4_m1_t5s_aug_r20_te_el025'
#    --work_dir './work_dirs/ga_myrpn_x101_voc_embsize1024_256_alpha015_lr000125-34-40-44_CE_ind1_beta004_aug_r20_t5s'
#    --work_dir './work_dirs/ga_dml_rpn2_x101_vocnovel1_1s_r40_embsize1024_256_alpha015_lr000125-34-40-44_CE_ind1_beta004_tloss4_te_el025'
#    --work_dir './work_dirs/ga_dml_rpn2_x101_vocbase1_embsize1024_256_alpha015_lr00025x4_8_12_14_CE_ind1_beta004_tloss4_te_el025'
#    --work_dir './work_dirs/ga_dml_rpn2_x101_coco_n10s_r15_embsize1024_256_alpha015_lr000125-34-40-44_CE_ind1_beta004_tloss4_te_el025'
CUDA_VISIBLE_DEVICES=4 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --work_dir './work_dirs/test'
#    --work_dir './work_dirs/ga_dml_rpn2_x101_cocobase2noval_n30s_r5_embsize1024_256_alpha015_lr0000125x1-34-40-44_CE_ind1_beta004_tloss4_te_el025'
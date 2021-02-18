#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2
PORT=${PORT:-87500}
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir 'work_dirs/retina_dml2_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_aug_standard1_10shot_r10_lr0001x2x1_warm1000_12_16_18_ind1_1_1'
#    --work_dir 'work_dirs/retina_dml2_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_base1_r1_lr00025x2x1_10_14_16_ind1_1'
#    --work_dir 'work_dirs/ga_retina_dml7_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_base1_r1_lr00025x2x2_14_18_20_ind4_1'
#    --work_dir 'work_dirs/retina_drt_s2_fpn_voc_aug_standard1_3shot_r20_lr00025x2x1_warm1000_12_16_18_ind1_1_1'
#    --work_dir 'work_dirs/plot_ga_retina_dml3_fpn_epoch_0'
#    --work_dir 'work_dirs/ga_retina_dml3_fpn_emb256_32_alpha015_le10_CE_nratio3_coco_aug_standard_10shot_r10_lr0001x1x2_14_18_20_ind1_1_1'
#    --work_dir 'work_dirs/retina_s2_fpn_voc_base1_r1_lr00125x2x4_8_11_12_ind1_1'
#    --work_dir 'work_dirs/ga_retina_dml3_s2_fpn_emb256_256_alpha015_le10_CE_nratio3_voc_aug_standard1_10shot_r10_lr00025x2x1_warm1000_12_16_18_ind1_1_1_direct'
#    --work_dir 'work_dirs/ga_retina_dml3_fpn_emb256_128_alpha015_le10_CE_nratio3_coco2voc_aug_standard_10shot_r10_lr0001x1x2_12_16_18_ind1_1_1_init_rep'
#    --work_dir 'work_dirs/retina_s4_fpn_voc_aug_standard1_1shot_r30_lr00025x2x1_warm1000_12_16_18_ind1_1_1'
#    --work_dir 'work_dirs/retina_drt_s4_fpn_voc_aug_standard1_5shot_r15_lr00025x2x1_warm1000_12_16_18_ind1_1_1'
#    --work_dir 'work_dirs/ga_retina_dml3_fpn_emb256_32_alpha015_le10_CE_nratio3_coco_base_r1_lr00025x2x2_16_22_24_ind1_1'
#    --work_dir 'work_dirs/retina_drt_s4_fpn_voc_base1_r1_lr00125x2x2_8_11_12_ind1_1'
#    --work_dir 'work_dirs/retina_fpn_voc_aug_standard1_3shot_r20_lr001x2x1_warm1000_12_16_18_ind1_1_1'
#    --work_dir 'work_dirs/ga_retina_dml3_s2_fpn_emb256_128_alpha003_le10_CE_nratio3_voc_base1_r1_lr00025x2x2_10_14_16_ind1_1'
#    --work_dir 'work_dirs/ga_retina_voc'
#    --work_dir 'work_dirs/ga_retina_dml7_s2_fpn_emb256_128_alpha015_le10_CE_nratio3_voc_base1_r1_lr00025x2x2_14_18_20_ind4_1'
#    --work_dir 'work_dirs/ga_retina_dml3_fpn_emb256_64_alpha015_le10_CE_nratio10_coco_aug_standard_10shot_r10_lr0001x2x2_14_18_20_ind2_1_1/'
#    --work_dir 'work_dirs/ga_retina_dml3_fpn_emb256_64_alpha015_le10_CE_nratio3_coco_base_r1_lr00025x2x2_20_24_26_ind3_1'
#!/usr/bin/env bash

DATASET='voc'
MODEL_NAME=ga_retina_dml14_voc_split1
PARAMETER=cls_n_dis_n_a/att_alpha_03

CONFIG_PATH='configs/few_shot/'$DATASET'/'$MODEL_NAME'/'$PARAMETER'/'
WORK_DIR_BASE='work_dirs/'$MODEL_NAME'/'$PARAMETER'/'
PORT=${PORT:-47500}
PYTHON=${PYTHON:-"python"}

CONFIG=$CONFIG_PATH'base.py'
GPUS=4
OMP_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES=4,5,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'base' && sleep 1s

GPU_ID=4,5,6,7
GPUS=4
CONFIG=$CONFIG_PATH'finetune.py'

for i in {1,2,3,5,10}
do
OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/finetune.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --shot $i \
    --work_dir $WORK_DIR_BASE$i'shot' \
    && sleep 5s
done

#GPU_ID=0,1,2,3,4,5,6,7
#GPUS=6
#CONFIG=$CONFIG_PATH'finetuneG.py'
#
#for i in {1,2,3,5,10}
#do
#  for j in {0..29}
#  do
#  OMP_NUM_THREADS=1 \
#  CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#      $(dirname "$0")/finetune.py $CONFIG --launcher pytorch ${@:3} \
#      --validate \
#      --shot $i \
#      --seedn $j \
#      --work_dir $WORK_DIR_BASE'G/'$i'shot/seed'$j\
#      && sleep 5s
#  done
#  sleep 3s
#done

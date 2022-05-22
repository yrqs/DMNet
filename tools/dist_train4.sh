#!/usr/bin/env bash

sleep 4h;

ps -ef | grep mmdetection | awk '{print $2}' | xargs kill -9;

DATASET='voc'
MODEL_NAME=frcn_r101_voc
PARAMETER=torchvision/fs_bbox_head3/1000_600/split2

CONFIG_PATH='configs/few_shot/'$DATASET'/'$MODEL_NAME'/'$PARAMETER'/'
WORK_DIR_BASE='work_dirs/'$MODEL_NAME'/'$PARAMETER'/'
PORT=${PORT:-27500}
PYTHON=${PYTHON:-"python"}

GPU_ID=0,1,2,3,4,5,6,7
GPUS=8

CONFIG=$CONFIG_PATH'base.py'
OMP_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --work_dir $WORK_DIR_BASE'base' && sleep 20s;

CONFIG=$CONFIG_PATH'finetuneD.py'
for i in {1,2,3,5,10}
do
OMP_NUM_THREADS=2 \
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/finetune.py $CONFIG --launcher pytorch ${@:3} \
    --validate \
    --shot $i \
    --work_dir $WORK_DIR_BASE'D/'$i'shot' \
    && sleep 20s
done;

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

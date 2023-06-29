#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}
CONFIG=$1
GPUS=$2
PORT=${PORT:-27500}

for shot in 1 2 3 5 10
do
  OMP_NUM_THREADS=4 \
  CUDA_VISIBLE_DEVICES=4,5,6,7 $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
      $(dirname "$0")/finetune.py $CONFIG --launcher pytorch ${@:3} \
      --shot $shot \
      --work_dir 'work_dirs/dmnet_voc_finetune1'
done

#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=3 python tools/test.py $CONFIG $CHECKPOINT \
  --show
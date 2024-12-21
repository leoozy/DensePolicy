#!/bin/bash

MODEL_PATH="/storage/home/westlakeLab/zhangjunlei/models/Meta-Llama-3.1-8B-Instruct/llama3.1_8b_intruct/Meta-Llama-3.1-8B-Instruct/"
SERVED_MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
#MODEL_PATH=/storage/home/westlakeLab/zhangjunlei/models/Mistral-7B-Instruct-v0.1/mistralai/Mistral-7B-Instruct-v0.1
#SERVED_MODEL_NAME="Mistral-7B-v0.1"
PORT_BASE=8000
TOTAL_GPUS=8
machine=07
seed=0

# 启动多个进程
for ((i=0; i<TOTAL_GPUS; i++))
do

  LOG_FILE=log/${machine}server_seed${seed}_$i.txt
  PORT=$((PORT_BASE + i))
  echo "Starting process on GPU $i with PORT $PORT"

  CUDA_VISIBLE_DEVICES=$i nohup vllm serve $MODEL_PATH \
    --port $PORT \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size 1 \
    --dtype auto --seed ${seed} > $LOG_FILE 2>&1 &
done
echo "All processes started. Logs are in the log directory."

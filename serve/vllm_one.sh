#MODEL_PATH="/storage/home/westlakeLab/zhangjunlei/models/Meta-Llama-3.1-8B-Instruct/llama3.1_8b_intruct/Meta-Llama-3.1-8B-Instruct/"
#SERVED_MODEL_NAME="Meta-Llama-3.1-8B-Instruct"
#MODEL_PATH=/storage/home/westlakeLab/zhangjunlei/models/Meta-Llama-3.1-70B-Instruct
#SERVED_MODEL_NAME="Meta-Llama-3.1-70B-Instruct"
MODEL_PATH=/storage/home/westlakeLab/zhangjunlei/models/Mistral-7B-v0.1/Mistral-7B-v0.1
SERVED_MODEL_NAME="Mistral-7B-v0.1"
PORT=8000
machine=14
seed=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
LOG_FILE=log/${machine}server_6.txt
  CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL_PATH \
    --port $PORT \
    --served-model-name $SERVED_MODEL_NAME \
    --tensor-parallel-size 1 \
    --dtype auto \
    --seed ${seed}
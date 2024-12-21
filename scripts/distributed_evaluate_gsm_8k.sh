#!/bin/bash

# 设置变量
MODEL_DIR=Meta-Llama-3.1-8B-Instruct
#Session_prefix=Llama_g_eo_42
#MODEL_DIR=Mistral-7B-v01
Session_prefix=gsm_baseline
TOTAL_DIS=8
LOG_DIR="log"
temp1=1.0
temp2=0.3
th=-0.1
seed=0
aim="gsm8k_${MODEL_DIR}_${temp1}_${temp2}_th${th}_seed${seed}_baseline_tmp03"

# 创建日志目录
mkdir -p "$LOG_DIR/$aim"

# 设置端口映射表
declare -A PORT_MAP=([0]=8000 [1]=8001 [2]=8002 [3]=8003 [4]=8004 [5]=8005 [6]=8006 [7]=8007)

# 设置 GPU 映射表
declare -A GPU_MAP=([0]=0 [1]=0 [2]=0 [3]=0)


# 第一部分：创建 tmux 会话
for ((DIS_ID=0; DIS_ID<TOTAL_DIS; DIS_ID++))
do
    SESSION_NAME="${Session_prefix}_$DIS_ID"
    if ! tmux has-session -t $SESSION_NAME 2>/dev/null; then
        echo "Creating session $SESSION_NAME"
        tmux new-session -d -s $SESSION_NAME
       # tmux send-keys -t $SESSION_NAME "export PATH=\"/root/anaconda3/bin/:$PATH\"" Enter
        tmux send-keys -t $SESSION_NAME "source activate rs" Enter
      #  tmux send-keys -t $SESSION_NAME "tmux set mouse on" Enter
        tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=$GPU" Enter
        tmux send-keys -t $SESSION_NAME "export TOKENIZERS_PARALLELISM=false" Enter
    fi
done

# 第二部分：在 tmux 会话中执行命令
for ((DIS_ID=0; DIS_ID<TOTAL_DIS; DIS_ID++))
do
    SESSION_NAME="${Session_prefix}_$DIS_ID"
    PORT=${PORT_MAP[$DIS_ID]}
    GPU=${GPU_MAP[$DIS_ID]}
    LOG_FILE=${LOG_DIR}/${aim}/processor_${DIS_ID}
    echo "OUTPUT log to ${LOG_FILE}"
    tmux send-keys -t $SESSION_NAME "cd ~/code/densepolicy && python downstream_test.py \
        --aim ${aim} \
        --dataset_name gsm8k \
        --planning_method codeact_agent \
        --max_turn 10 \
        --in_context_number 8 \
        --model_name $MODEL_DIR \
        --host $PORT \
        --exp_id 0 \
        --distributed_test \
        --distributed_id $DIS_ID \
        --distributed_number $TOTAL_DIS \
        --resume \
        --max_steps 200 \
        --resume_from_merge  2>&1 | tee $LOG_FILE " Enter
done

echo "所有进程已在单独的 tmux 会话中启动。"

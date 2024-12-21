PORT=8003
export TOKENIZERS_PARALLELISM=false
#MODEL_DIR=Qwen2.5-7B-Instruct
MODEL_DIR=Meta-Llama-3.1-8B-Instruct
dataset_name=gsm8k
aim=baseline
LOG_FILE="log/${dataset_name}_${MODEL_DIR}_${aim}.txt"
nohup python downstream_test.py  --aim ${aim} --dataset_name gsm8k --planning_method codeact_agent --max_turn 10  --in_context_number 8 --model_name $MODEL_DIR --host $PORT --exp_id 0 > $LOG_FILE 2>&1 &
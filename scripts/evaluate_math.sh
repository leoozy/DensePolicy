PORT=8004
export TOKENIZERS_PARALLELISM=false
MODEL_DIR=Qwen2.5-7B-Instruct
#MODEL_DIR=Meta-Llama-3.1-8B-Instruct
dataset_name=math
aim=search_without_confidential
LOG_FILE="log/${dataset_name}_${MODEL_DIR}_${aim}.txt"
python downstream_test.py  --aim ${aim} --dataset_name ${dataset_name} --planning_method codeact_agent --max_steps 200 --in_context_number 1 --model_name $MODEL_DIR --host $PORT --max_turn 10 --exp_id 0
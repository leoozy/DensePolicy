#!/bin/bash

#if [ "$#" -ne 5 ]; then
#    echo "Usage: ./launch_server_and_agent.sh TEST_ID CUDA_VISIBLE_DEVICES PORT MODEL_DIR TASK"
#    exit 1
#fi

TEST_ID=$1
CUDA_VISIBLE_DEVICES=$2
MODEL_DIR=$3
TASK=$4
METHOD=$5
CUSTOM_PORT=$6
CUSTOM_DIS_ID=$7
CUSTOM_TOTAL_DIS=$8

# Convert CUDA_VISIBLE_DEVICES to an array
IFS=',' read -r -a cuda_array <<< "$CUDA_VISIBLE_DEVICES"

# Get the length of the array
NUM_DEVICE=${#cuda_array[@]}

MAIN_DIR='.'


# Launch AgentGym server
case $TASK in
  'babyai'|'maze'|'wordle'|'sciworld'|'sqlgym'|'textcraft'|'webshop')
    # Compute the Port
    ENV_PORT=$((46001+TEST_ID))

    if ! tmux has-session -t environment$TEST_ID 2>/dev/null; then
      tmux new-session -d -s environment$TEST_ID
      tmux send-keys -t environment$TEST_ID 'export PATH="/root/anaconda3/bin/:$PATH"' Enter
      tmux send-keys -t environment$TEST_ID "tmux set mouse on" Enter
    fi

    echo "Launching AgentGym environment with port $ENV_PORT in tmux window environment$TEST_ID"

    case $TASK in
      'babyai')
        tmux send-keys -t "environment$TEST_ID" "conda deactivate ; conda activate agentenv-babyai ; babyai --host 0.0.0.0 --port $ENV_PORT" Enter
        ;;
      'maze'|'wordle')
        tmux send-keys -t "environment$TEST_ID" "conda deactivate ; conda activate agentenv-lmrlgym ; lmrlgym --host 0.0.0.0 --port $ENV_PORT" Enter
        ;;
      'sciworld')
        tmux send-keys -t "environment$TEST_ID" "conda deactivate ; conda activate agentenv-sciworld ; sciworld --host 0.0.0.0 --port $ENV_PORT" Enter
        ;;
      'sqlgym')
        tmux send-keys -t "environment$TEST_ID" "conda deactivate ; conda activate agentenv-sqlgym ; export AGENTENV_SQLGYM_BIRD_PATH='$MAIN_DIR/LLM-Agent-Eval/dataset/AgentGym/agentenv-sqlgym/bird' ; sqlgym --host 0.0.0.0 --port $ENV_PORT" Enter
        ;;
      'textcraft')
        tmux send-keys -t "environment$TEST_ID" "conda deactivate ; cd '$MAIN_DIR/LLM-Agent-Eval/dataset/AgentGym/agentenv-textcraft' ; conda activate agentenv-textcraft ; textcraft --host 0.0.0.0 --port $ENV_PORT" Enter
        ;;
      'webshop')
        tmux send-keys -t "environment$TEST_ID" "conda deactivate ; conda activate agentenv-webshop ; webshop --host 0.0.0.0 --port $ENV_PORT" Enter
        ;;
      *)
        ;;
    esac
    echo "AgentGym environment server has launched"
    ;;
  *)
esac



## Start agentTEST_ID tmux window
#if ! tmux has-session -t agent$TEST_ID 2>/dev/null; then
#  tmux new-session -d -s agent$TEST_ID
#  tmux send-keys -t agent$TEST_ID 'export PATH="/root/anaconda3/bin/:$PATH"' Enter
#  tmux send-keys -t agent$TEST_ID "source activate agent_env" Enter
#  tmux send-keys -t agent$TEST_ID "tmux set mouse on" Enter
#fi
#
#tmux send-keys -t agent$TEST_ID "cd $MAIN_DIR/LLM-Agent-Eval" Enter


# Compute distributed id

# Compute the Port
if [ -z "$CUSTOM_DIS_ID" ]; then
    DIS_ID=$TEST_ID
else
    DIS_ID=$CUSTOM_DIS_ID
fi
echo "Distributed ID $DIS_ID"

# Compute the total distributed number
if [ -z "$CUSTOM_TOTAL_DIS" ]; then
    TOTAL_DIS=8
else
    TOTAL_DIS=$CUSTOM_TOTAL_DIS
fi
echo "Total Distributed ID $TOTAL_DIS"


echo "Evaluating on task $TASK in TMUTEST_ID agent$TEST_ID"
# Execute script in agentTEST_ID tmux window based on TASK
if [ "$TASK" = "gsm8k" ]; then
  if [ "$METHOD" = "cot" ]; then
      tmux send-keys -t agent$TEST_ID "./scripts/eval/gsm8k/gsm8k.sh $SAVE_DIR/$MODEL_DIR $MODEL_DIR $PORT cot $KILL_COMMAND" Enter
  elif [ "$METHOD" = "codeact_agent" ]; then
      tmux send-keys -t agent$TEST_ID "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python downstream_test.py --dataset_name gsm8k --planning_method codeact_agent --in_context_number 8 --model_name $MODEL_DIR --host $PORT --max_turn 10 --exp_id 0 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND" Enter
  elif [ "$METHOD" = "codeact_agent_tree" ]; then
      tmux send-keys -t agent$TEST_ID "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python downstream_test.py --dataset_name gsm8k --planning_method codeact_agent_tree --in_context_number 8 --model_name $MODEL_DIR --host $PORT --max_steps 32 --exp_id 0 --contrastive_method temperature --max_sample_number 256 --window_size 16 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND" Enter
  fi

elif [ "$TASK" = "math" ]; then
  if [ "$METHOD" = "codeact_agent" ]; then
      tmux send-keys -t agent$TEST_ID "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python downstream_test.py --dataset_name math --planning_method codeact_agent --in_context_number 1 --model_name $MODEL_DIR --host $PORT --max_turn 10 --exp_id 0 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND" Enter
  elif [ "$METHOD" = "codeact_agent_tree" ]; then
      echo "running codeact_agent_tree"
      tmux send-keys -t agent$TEST_ID "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python downstream_test.py --dataset_name math --planning_method codeact_agent_tree --in_context_number 1  --model_name $MODEL_DIR --host $PORT --max_steps 8 --exp_id 0 --contrastive_method temperature --max_sample_number 512 --window_size 8 --base 10 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge  $KILL_COMMAND" Enter
  fi

elif [ "$TASK" = "miniwob" ]; then
  if [ "$METHOD" = "cot" ]; then
      tmux send-keys -t agent$TEST_ID "./scripts/eval/miniwob++/miniwob++.sh $SAVE_DIR/$MODEL_DIR $CKPT_DIR/$MODEL_DIR $PORT $KILL_COMMAND" Enter
  fi
elif [[ $TASK == "alfworld"* ]]; then
  if [ "$METHOD" = "codeact_agent" ]; then
    tmux send-keys -t agent$TEST_ID "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python downstream_test.py --dataset_name $TASK --planning_method codeact_agent --model_name $MODEL_DIR --host $PORT --max_steps 30 --exp_id 0  --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND" Enter
  elif [ "$METHOD" = "codeact_agent_tree" ]; then
    tmux send-keys -t agent$TEST_ID "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python downstream_test.py --dataset_name $TASK --planning_method codeact_agent_tree --model_name $MODEL_DIR --host $PORT --max_steps 30 --exp_id 0 --contrastive_method temperature --max_sample_number 512 --window_size 8 --base 8 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge  $KILL_COMMAND" Enter
  fi
elif [ "$TASK" = "m3tooleval" ]; then
  if [ "$METHOD" = "codeact_agent" ]; then
    tmux send-keys -t agent$TEST_ID "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python downstream_test.py --dataset_name $TASK --planning_method codeact_agent --model_name $MODEL_DIR --host $PORT --max_steps 10 --exp_id 0 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND" Enter
  elif [ "$METHOD" = "codeact_agent_tree" ]; then
    tmux send-keys -t agent$TEST_ID "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python downstream_test.py --dataset_name $TASK --planning_method codeact_agent_tree --model_name $MODEL_DIR --host $PORT --max_steps 10 --exp_id 0 --contrastive_method temperature --max_sample_number 256 --window_size 5 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND" Enter
  fi
elif [ "$TASK" = "babyai" ]; then
  if [ "$METHOD" = "codeact_agent" ]; then
    tmux send-keys -t agent$TEST_ID "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python downstream_test.py --dataset_name $TASK --dataset_port_id $ENV_PORT --planning_method codeact_agent --model_name $MODEL_DIR --host $PORT --max_steps 30 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge  $KILL_COMMAND" Enter
  elif [ "$METHOD" = "codeact_agent_tree" ]; then
    tmux send-keys -t agent$TEST_ID "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python downstream_test.py --dataset_name $TASK --dataset_port_id $ENV_PORT --planning_method codeact_agent_tree --model_name $MODEL_DIR --host $PORT --max_steps 30 --contrastive_method temperature --max_sample_number 256 --window_size 15 --distributed_test --distributed_id $DIS_ID --distributed_number $TOTAL_DIS --resume --resume_from_merge $KILL_COMMAND" Enter
  fi
fi
PORT=8000
CUDA_VISIBLE_DEVICES=1 python downstream_test.py --dataset_name math --model_name Meta-Llama-3.1-8B-Instruct --planning_method codeact_agent_tree --in_context_number 1  --host $PORT --max_steps 8 --exp_id 0 --contrastive_method temperature --max_sample_number 512 --window_size 8 --base 10
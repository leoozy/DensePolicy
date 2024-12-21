python post_process_training_data.py --process_result --data_dir eval_results/codeact_agent_tree.Meta-Llama-3.1-8B-Instruct/temperature_512_10_8/  --dataset_file_name_list math.None_split_0.0.None.csv --data_sub_dir math --dataset_file_name math.None_split_0.0.None.csv
python post_process_training_data.py --merge_all_pairs
python prepare_dataset.py
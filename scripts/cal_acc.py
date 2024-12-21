import csv
import pdb
import os
import pandas as pd

result = []
data_path = "/storage/home/westlakeLab/zhangjunlei/code/densepolicy/eval_results/codeact_agent.Meta-Llama-3.1-8B-Instruct.gsm8k_Meta-Llama-3.1-8B-Instruct_1.0_0.3_th-0.1_seed0_baseline_tmp03/200.10.8_False"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
print(f"calculating the acc for files in {data_path}")
print(f"The number of files is {len(csv_files)}: {csv_files}")
for csv_file in csv_files:
    df = pd.read_csv(os.path.join(data_path, csv_file))
    for score in df["score"]:
        result.append(score)
    print(f"Loaded {csv_file}")
    print(f"Current length: {len(result)}")
    print(f"Current mean: {sum(result) / len(result)}")
print(f"Total length: {len(result)}")
print(f"Total mean: {sum(result) / len(result)}")

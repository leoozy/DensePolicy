import pandas as pd
import os

# 设置文件夹路径
folder_path = '/storage/home/westlakeLab/zhangjunlei/code/densepolicy/eval_results/codeact_agent.Meta-Llama-3.1-8B-Instruct.MATH_Meta-Llama-3.1-8B-Instruct_1.0_0.3_th-0.1_seed0_search_max/200.10.1_False'

# 获取文件夹下所有的CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 创建一个空的DataFrame用于合并
combined_df = pd.DataFrame()

# 逐个读取CSV文件，并合并到combined_df
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    combined_df = pd.concat([combined_df, df], ignore_index=True)
output_name = folder_path.split('/')[-2]
# 保存合并后的数据到一个新的CSV文件
combined_df.to_csv(f'{folder_path}/{output_name}_results.csv', index=False)

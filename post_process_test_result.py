import argparse
import copy
import os

import numpy as np
import pandas as pd
import shutil

from model.codeact_agent_tree.codeact_agent_tree import Contrast_Pair_MCTree
from model.codeact_agent.codeact_agent import codeact_agent


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--merge_multi_test_result", action="store_true")
parser.add_argument("--merge_distributed_result", action="store_true")
parser.add_argument("--see_score", action="store_true")
parser.add_argument("--see_history", action="store_true")
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--dataset_file_name_list", type=str, nargs='+')

args, left = parser.parse_known_args()

if args.merge_multi_test_result:
    file_path_list = [
        # 'eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct/temperature_256_3/math.None_split_0.0.None.csv',
        'eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct/temperature_512_4_30/math.None_split_0.0.None.csv',
        # 'eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct/temperature_512_10_8/math.None_split_0.0.None.csv'
    ]
    df_list = []
    for file_path in file_path_list:
        df_in = pd.read_csv(file_path, usecols=['index'])
        df_list.append(df_in)
    df = pd.concat(df_list, ignore_index=True)
    print(len(df['index'].unique()))
    df.to_csv('eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct/temperature_512_10_8/math_resume_index.csv')


def preprocess_dataset(dataset_pd):
    def eval_nona(item):
        if pd.isna(item):
            return item
        else:
            return eval(item)

    for key in dataset_pd.columns:
        try:
            dataset_pd[key] = dataset_pd[key].apply(eval_nona)
        except Exception as e:
            pass

    return dataset_pd


if args.merge_distributed_result:
    remove_repeat = False

    distributed_subdir = os.path.join(args.data_dir, 'distributed_result')
    if not os.path.exists(distributed_subdir):
        os.makedirs(distributed_subdir)

    for dataset_file_name in args.dataset_file_name_list:
        file_path = os.path.join(args.data_dir, dataset_file_name.format('None'))
        if args.merge_distributed_result:
            dis_path = os.path.join(args.data_dir, dataset_file_name)
            dis_distributed_path = os.path.join(args.data_dir, 'distributed_result', dataset_file_name)
            res_list = []
            for i in range(1000):
                # index is not used (As the file saved)
                if os.path.exists(dis_path.format(i)):
                    moved_ind = i
                    while os.path.exists(dis_distributed_path.format(moved_ind)):
                        moved_ind += 1
                    print('Moving testing file {} into distributed_result {}'.format(i, moved_ind))
                    shutil.move(dis_path.format(i), dis_distributed_path.format(moved_ind))
                if os.path.exists(dis_distributed_path.format(i)):
                    res = pd.read_csv(dis_distributed_path.format(i), index_col=0
                                      )
                    res_list.append(res)
            print('{} distributed files found'.format(len(res_list)))
            merged_res = pd.concat(res_list, ignore_index=True)
            data_count = merged_res['index'].value_counts()
            print('{} data in total'.format(len(data_count)))
            repeated_data = [key for key in data_count.index if data_count[key] > 1]
            print('{} repeated data'.format(len(repeated_data)))
            for index, value in data_count.items():
                if value > 1:
                    print('Warning! Data {} is tested {} times'.format(index, value))
                    if remove_repeat:
                        lines = merged_res[merged_res['index'] == index]
                        lines_pair_num = copy.deepcopy(lines)


                        def compute_pair_num(res_cur):
                            res_cur = eval(res_cur)
                            mctree = Contrast_Pair_MCTree(3, 10)
                            for index_1 in res_cur:
                                mctree.update_score(index_1, res_cur[index_1], no_info=True)
                            return mctree.get_contrast_pair_num()


                        tem = lines_pair_num['res_cur'].apply(compute_pair_num)
                        lines_pair_num.loc[:, 'pair_num'] = tem
                        print(lines_pair_num['pair_num'])
                        non_max_index = lines_pair_num.drop(lines_pair_num['pair_num'].idxmax()).index
                        print('Dropping {}'.format(non_max_index))
                        merged_res = merged_res.drop(non_max_index)

            print('{} line in total'.format(len(merged_res)))
            merged_res.to_csv(file_path)

            print(100 * merged_res['score'].mean())

if args.see_score or args.see_history:
    dataset_file_name_list = [item.format('None') for item in args.dataset_file_name_list]

    for dataset_file_name in dataset_file_name_list:
        all_turns = []
        all_scores = []
        dis_path = os.path.join(args.data_dir, dataset_file_name)
        res = pd.read_csv(dis_path, index_col=0
                          )
        res = preprocess_dataset(res)
        print(len(res))
        print(dataset_file_name.split('.')[0])

        if args.see_score:
            for turn in range(res['turns'].max() + 1):
                score_mean = 100 * ((res['score'] == 1) & (res['turns'] <= turn + 1)).mean()
                print(turn + 1, score_mean)
                all_turns.append(turn)
                all_scores.append(score_mean)
                # print(np.mean(all_scores))

                if 'task_type' in res.columns:
                    sub_task_scores = []
                    all_sub_task = res['task_type'].unique()
                    ordered_subtasks = ['put', 'examine', 'clean', 'cool', 'heat', 'puttwo']
                    assert set(all_sub_task) == set(ordered_subtasks)
                    for sub_task in ordered_subtasks:
                        sub_task_score = 100 * res[res['task_type'] == sub_task]['score'].mean()
                        sub_task_scores.append(sub_task_score)
                        print(sub_task, sub_task_score)
                    print(ordered_subtasks)
                    print(sub_task_scores)
                    print(np.mean(sub_task_scores))

            print(all_turns)
            print(all_scores)

        else:
            for ind, line in res.iterrows():
                codeact_agent.print_history(line['history'])
                print(line['target'])

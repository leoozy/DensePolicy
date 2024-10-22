import argparse
import copy
import fnmatch
import os
import shutil

import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
# import dask.dataframe as dd
from tqdm import tqdm

from sup_func.sup_func import printc

from model.codeact_agent_tree.codeact_agent_tree import Contrast_Pair_MCTree, MCValue_MCTree

# import torch

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--illustrate_trajectory", action="store_true")
parser.add_argument("--process_result", action="store_true")
parser.add_argument("--merge_all_pairs", action="store_true")
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--data_sub_dir", type=str, default=None)
parser.add_argument("--dataset_file_name", type=str, default=None)

args, left = parser.parse_known_args()

merge_distributed_trees = True
generate_pairs_pd = True
plot_hist = True
plot_tree = True

tree_policy = 'mc_value'
assert tree_policy in ['mc_value', 'contrast_pair']
base = 10

thre_max = 8
max_length = 30  # not matter for training data process
thre_value = 0.5


# CodeAct illustration

def see_history(history):
    for index, line in enumerate(history):
        step_id = index // 2
        if index == 0:
            printc("Instruction: {}".format(line['content']), 'yellow')
        else:
            if line['role'] == 'user':
                printc("State {}: {}".format(step_id, line['content']), 'blue')
            else:
                printc("Generation {}: {}".format(step_id, line['content']), 'green')


def remove_in_context_example(history):
    if 'Here are examples:' in history[0]['content']:
        last_example_start = 0
        for last_example_start in range(len(history) - 1, -1, -1):
            if history[last_example_start]['role'] == 'user' and "Now, let's move on to the next task." in \
                    history[last_example_start]['content']:
                break
        # print('Last example index: {}'.format(last_example_start))
        new_task_flag = "Now, let's move on to the next task. The instructions for this task are the same as the previous ones."
        assert new_task_flag in history[last_example_start]['content']
        goal = history[last_example_start]['content'].split(new_task_flag)[1]
        new_instruction = history[0]['content']
        new_instruction = new_instruction.split('Here are examples:')[0]
        new_instruction = new_instruction.strip() + '\n' + goal.strip()
        new_instruction_turn = copy.deepcopy(history[0])
        new_instruction_turn['content'] = new_instruction
        history_new = [new_instruction_turn] + history[(last_example_start + 1):]
        return history_new
    else:
        return history


if args.illustrate_trajectory:
    res = pd.read_csv(
        'eval_results/codeact_agent.Meta-Llama-3-8B-Instruct/gsm8k.None_split_0.0.None.csv')

    res['history'] = res['history'].apply(lambda x: eval(x))
    for history in res['history']:
        see_history(history)


def select_mc_value_pair(tree_pairs_pd):
    visit_num_thre_positive = tree_pairs_pd.apply(
        lambda line: min(math.floor(np.power(2, line['positive_avg_depth'])), thre_max), axis=1)

    visit_num_thre_negative = tree_pairs_pd.apply(
        lambda line: min(math.floor(np.power(2, line['negative_avg_depth'])), thre_max), axis=1)

    tree_pairs_pd['selected'] = (tree_pairs_pd['positive_visit_number'] >= visit_num_thre_positive) & (
            tree_pairs_pd['negative_visit_number'] >= visit_num_thre_negative) & (
                                        tree_pairs_pd['positive_value'] - tree_pairs_pd['negative_value'] >= thre_value)
    return tree_pairs_pd


# Tree Data PostProcess
if args.process_result:

    if not os.path.exists(os.path.join(args.data_dir, args.data_sub_dir)):
        os.makedirs(os.path.join(args.data_dir, args.data_sub_dir))

    assert '.None.csv' in args.dataset_file_name
    file_path = os.path.join(args.data_dir, args.data_sub_dir,
                             args.dataset_file_name.replace('.None.csv', '.merge.csv'))
    if merge_distributed_trees:
        if not os.path.exists(file_path):
            shutil.copyfile(os.path.join(args.data_dir, args.dataset_file_name), file_path)
    # else:
    #     raise ValueError(f"The file '{file_path}' already exists. Aborting copy operation.")

    res = pd.read_csv(file_path,
                      # nrows=20
                      )

    res['res_cur'] = res['res_cur'].apply(lambda x: eval(x))
    score_df = pd.DataFrame(list(res['res_cur']))
    # print(dict(score_df.mean()))

    res['history'] = res['history'].apply(lambda x: eval(x))
    history_df = pd.DataFrame(list(res['history']))

    if generate_pairs_pd:
        tree_pairs = []
        for i in tqdm(range(len(score_df))):
            tree_pairs_cur = []
            key_list = list(score_df.keys()[~score_df.iloc[i].isna()])
            cur_data_max_length = np.max([len(item) for item in key_list])
            print('Num Testing: {}'.format(len(key_list)))
            print('Max length {}'.format(cur_data_max_length))

            mctree = Contrast_Pair_MCTree(base, max_length)
            mc_value_tree = MCValue_MCTree(base, max_length)
            for start, index_1 in tqdm(enumerate(key_list)):
                mctree.update_score(index_1, score_df[index_1][i], no_info=True)
                history_without_ic = remove_in_context_example(history_df[index_1][i])
                if len(history_without_ic) == len(index_1) * 2 + 1:
                    mctree.update_node_score(index_1, score_df[index_1][i],
                                             history=history_without_ic)
                    mc_value_tree.update_node_score(index_1, score_df[index_1][i], history=history_without_ic,
                                                    no_info=True)
                else:
                    print(
                        'Warning! history length {} does not match with index length {}'.format(len(history_without_ic),
                                                                                                len(index_1)))
            print('Num Pairs by mctree: {}'.format(mctree.get_contrast_pair_num()))
            mc_value_tree.check_scores()
            res_mv_value_tree = mc_value_tree.get_contrast_pairs()
            if res_mv_value_tree['score_diff'].max() > 0:
                print(1)
            num_pairs = mc_value_tree.get_contrast_pair_num()
            assert num_pairs == res_mv_value_tree['pass_filter'].sum()
            print('Num Pairs by mcvaluetree unfiltered: {}'.format(len(res_mv_value_tree)))
            print('Num Pairs by mcvaluetree: {}'.format(num_pairs))

            if tree_policy == 'contrast_pair':
                for start, index_1 in enumerate(key_list):
                    for index_2 in key_list[(start + 1):]:
                        # if (not pd.isna(score_df[index_1][i])) and (not pd.isna(score_df[index_2][i])) and \
                        #         score_df[index_1][
                        #             i] != score_df[index_2][i]:
                        flag, contrast_step = Contrast_Pair_MCTree.is_contrast_index(index_1, index_2)
                        if flag and score_df[index_1][
                            i] != score_df[index_2][i]:
                            if score_df[index_1][i] > 0:
                                positive_run_index = index_1
                                negative_run_index = index_2
                            else:
                                positive_run_index = index_2
                                negative_run_index = index_1

                            positive_score = score_df[positive_run_index][i]
                            positive_history = history_df[positive_run_index][i]
                            negative_score = score_df[negative_run_index][i]
                            negative_history = history_df[negative_run_index][i]
                            for turn in positive_history:
                                if turn['role'] == 'assistant':
                                    turn['content'] = turn['content'].strip()
                            for turn in negative_history:
                                if turn['role'] == 'assistant':
                                    turn['content'] = turn['content'].strip()
                            assert positive_history[:(1 + 2 * contrast_step)] == negative_history[
                                                                                 :(1 + 2 * contrast_step)]
                            positive_node_mcnode = mctree.node_MCTree[
                                positive_run_index[:(1 + 2 * contrast_step) + 1]]
                            positive_visit_number = positive_node_mcnode['num_node_explore']
                            positive_value = positive_node_mcnode['node_score']
                            negative_node_mcnode = mctree.node_MCTree[
                                negative_run_index[:(1 + 2 * contrast_step) + 1]]
                            negative_visit_number = negative_node_mcnode['num_node_explore']
                            negative_value = negative_node_mcnode['node_score']

                            if positive_history[1 + 2 * contrast_step] != negative_history[1 + 2 * contrast_step]:
                                tree_pairs_cur.append(
                                    [res['index'][i], res['input'][i], res['target'][i], contrast_step,
                                     positive_score, positive_run_index, positive_history, positive_visit_number,
                                     positive_value,
                                     negative_score,
                                     negative_run_index,
                                     negative_history, negative_visit_number, negative_value])
                            else:
                                print('Warning: contrast step is the same')
                print('Num Pairs without Same contrast step: {}'.format(len(tree_pairs_cur)))
                tree_pairs += tree_pairs_cur
            else:
                for index_1 in mc_value_tree.values:
                    for index_2 in mctree.same_parent_node_for_traversal(index_1):
                        if index_2 in mc_value_tree.values:
                            if abs(mc_value_tree[index_1]['score'] - mc_value_tree[index_2][
                                'score']) > 0:

                                if mc_value_tree[index_1]['score'] > mc_value_tree[index_2][
                                    'score']:
                                    positive_run_index = index_1
                                    negative_run_index = index_2
                                else:
                                    positive_run_index = index_2
                                    negative_run_index = index_1

                                # positive_score = score_df[positive_run_index][i]
                                positive_history = mc_value_tree[positive_run_index]['history']
                                # negative_score = score_df[negative_run_index][i]
                                negative_history = mc_value_tree[negative_run_index]['history']
                                for turn in positive_history:
                                    if turn['role'] == 'assistant':
                                        turn['content'] = turn['content'].strip()
                                for turn in negative_history:
                                    if turn['role'] == 'assistant':
                                        turn['content'] = turn['content'].strip()
                                assert positive_history[:-1] == negative_history[:-1]

                                positive_node_mcnode = mc_value_tree[positive_run_index]
                                positive_visit_number = positive_node_mcnode['num_explore']
                                positive_value = positive_node_mcnode['score']
                                positive_avg_depth = positive_node_mcnode['score_depth_avg']

                                negative_node_mcnode = mc_value_tree[negative_run_index]
                                negative_visit_number = negative_node_mcnode['num_explore']
                                negative_value = negative_node_mcnode['score']
                                negative_avg_depth = negative_node_mcnode['score_depth_avg']

                                if positive_history[-1] != negative_history[-1]:
                                    tree_pairs_cur.append(
                                        [res['index'][i], cur_data_max_length, res['input'][i], res['target'][i],
                                         len(index_1) - 1,
                                         1, positive_run_index, positive_history, positive_visit_number,
                                         positive_value, positive_avg_depth,
                                         0,
                                         negative_run_index,
                                         negative_history, negative_visit_number, negative_value, negative_avg_depth])
                                else:
                                    print('Warning: contrast step is the same')
                print('Num Pairs without Same contrast step: {}'.format(len(tree_pairs_cur)))
                tree_pairs += tree_pairs_cur

        tree_pairs_pd = pd.DataFrame(tree_pairs,
                                     columns=['data_id', 'cur_data_max_length', 'input', 'target', 'contrast_step',
                                              'positive_score',
                                              'positive_run_index', 'positive_history', 'positive_visit_number',
                                              'positive_value', 'positive_avg_depth',
                                              'negative_score', 'negative_run_index', 'negative_history',
                                              'negative_visit_number', 'negative_value', 'negative_avg_depth'])

        tree_pairs_pd = select_mc_value_pair(tree_pairs_pd)
        print('{} pair selected'.format(tree_pairs_pd['selected'].sum()))
        print("{} pairs created".format(len(tree_pairs_pd)))
        tree_pairs_pd.to_csv(
            file_path + '{}_{}_{}_tree_pairs_result.csv'.format(tree_policy, thre_max,
                                                                thre_value))
    else:
        tree_pairs_pd = pd.read_csv(
            file_path + '{}_{}_{}_tree_pairs_result.csv'.format(tree_policy, thre_max,
                                                                thre_value))

    tree_pairs_pd = tree_pairs_pd[tree_pairs_pd['selected'] == True]
    # Create histogram
    if plot_hist:
        # Note that value_counts is for data_id, as one data can generate multiple pairs. So the list(size) is the pair number for each data
        pair_number_each_data = tree_pairs_pd['data_id'].value_counts()
        pair_number_each_data = pair_number_each_data.reindex(res['index'], fill_value=0)
        # Get the minimum and maximum count values, excluding zero
        min_count = pair_number_each_data[pair_number_each_data > 0].min() if len(
            pair_number_each_data[pair_number_each_data > 0]) > 0 else 1
        max_count = pair_number_each_data.max()
        # Create custom bins with a separate bin for zero
        bins = [-0.5, 0.5] + list(range(min_count, max_count + 2))
        plt.hist(list(pair_number_each_data), bins=bins, edgecolor='black', density=True)
        plt.xlabel('Contrastive pair number for one data')
        plt.ylabel('Frequency')
        plt.title('Total {} Pairs for {} Data'.format(len(tree_pairs_pd), len(res)))
        plt.savefig(file_path + 'freq_histogram.png')
        plt.close()

        test_num = tree_pairs_pd['data_id'].value_counts()

    # Create Tree Graph Figure
    if plot_tree:
        def plot_tree_for_data_id(data_id):
            example_tree_id = list(
                tree_pairs_pd[tree_pairs_pd['data_id'] == data_id]['positive_run_index']) + list(
                tree_pairs_pd[tree_pairs_pd['data_id'] == data_id]['negative_run_index'])
            example_tree_labels = list(
                tree_pairs_pd[tree_pairs_pd['data_id'] == data_id]['positive_score']) + list(
                tree_pairs_pd[tree_pairs_pd['data_id'] == data_id]['negative_score'])

            def build_tree(data, given_label):
                tree = nx.DiGraph()
                for row, row_label in zip(data, given_label):
                    for i, value in enumerate(row):
                        parent = row[:i]
                        child = row[:i + 1]
                        if child not in tree:
                            if len(child) == len(row):
                                label = row_label
                            else:
                                label = -1
                            tree.add_node(child, label=label)
                        if parent not in tree:
                            tree.add_node(parent, label=-1)
                        tree.add_edge(parent, child)
                return tree

            def plot_tree(tree):
                pos = nx.nx_agraph.graphviz_layout(tree, prog='dot')

                def get_color(label):
                    if label == 0:
                        return (1, 0.5, 0.5)  # lightred
                    elif label == 1:
                        return "lightgreen"
                    else:
                        return "lightblue"

                # Create a list of colors for each node in the graph based on their labels
                colors = [get_color(tree.nodes[node]['label']) for node in tree.nodes()]
                nx.draw(tree, pos, with_labels=False, node_size=100, node_color=colors, font_size=10)
                plt.savefig(file_path + 'tree_example_{}.png'.format(data_id))
                plt.close()

            tree = build_tree(example_tree_id, example_tree_labels)
            plot_tree(tree)


        tree_data_ids = tree_pairs_pd['data_id'].unique()
        for data_id in tqdm(np.random.choice(tree_data_ids, min(10, len(tree_data_ids)), replace=False)):
            plot_tree_for_data_id(data_id)

if args.merge_all_pairs:
    dataset_merge_pair_number = {'math': 22331, 'alfworld_cool': 665, 'alfworld_put': 1170, 'babyai': 908}
    # dataset_merge_sample_number = {'math': 347, 'alfworld_cool': 20, 'alfworld_put': 24, 'babyai': 19} # standard
    dataset_merge_sample_number = {'math': min(347 * 8, 2334), 'alfworld_cool': 20, 'alfworld_put': 24, 'babyai': 19}

    pairs_dataset_pre = []
    pairs_dataset = [
        [
            'eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct/temperature_512_4_30/math/math.None_split_0.0.merge.csvmc_value_8_0.2_tree_pairs_result.csv',
            'eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct/temperature_512_10_8/math/math.None_split_0.0.merge.csvmc_value_8_0.2_tree_pairs_result.csv',
            'eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct/temperature_256_3/math/math.None_split_0.0.merge.csvmc_value_8_30_1000_0.5_tree_pairs_result.csv'],
    ]
    if not pairs_dataset:
        for root, dirs, files in os.walk(args.data_dir):
            for file in files:
                if fnmatch.fnmatch(file, '*.merge.csv{}_*.csv'.format(tree_policy)) and (
                        len(pairs_dataset_pre) == 0 or file in pairs_dataset_pre):
                    file_path = os.path.join(root, file)
                    print(f"Found file: {file_path}")
                    pairs_dataset.append((file_path, file))
        pairs_dataset = sorted(pairs_dataset, key=lambda x: pairs_dataset_pre.index(x[1]))

    df_list = []
    task_list = []
    for ind, file_path_list in enumerate(pairs_dataset):
        df_load_list = []
        for file_path in file_path_list:
            df_in = pd.read_csv(file_path)
            df_load_list.append(df_in)
        df = pd.concat(df_load_list, ignore_index=True)
        df = df[df['selected'] == True]
        file = os.path.basename(file_path_list[0])
        task = file.split('.')[0]
        for file_path in file_path_list:
            file = os.path.basename(file_path)
            assert file.split('.')[0] == task
        df['task'] = task
        print(task)
        print('Pair number: ', len(df))
        print('Sample number: ', len(df['data_id'].unique()))
        if dataset_merge_sample_number:
            pair_number_each_data = df['data_id'].value_counts()
            sorted_data_id = sorted(pair_number_each_data.index, reverse=True, key=lambda x: pair_number_each_data[x])
            print({id: pair_number_each_data[id] for id in sorted_data_id})
            selected = sorted_data_id[:min(len(sorted_data_id), dataset_merge_sample_number[task])]
            ratio = dataset_merge_pair_number[task] / pair_number_each_data[selected].sum()
            df = df[df['data_id'].isin(selected)]
            if ratio < 1:
                df = df.groupby('data_id', group_keys=False).apply(lambda x: x.sample(frac=ratio, random_state=42))
            print('Pair number after selection: ', len(df))
            print('Sample number after selection: ', len(df['data_id'].unique()))
        df_list.append(df)
        task_list.append(task)

    merged_pairs_data = pd.concat(df_list, ignore_index=True)
    for task in merged_pairs_data['task'].unique():
        print(task)
        print('Pair number: ', len(merged_pairs_data[merged_pairs_data['task'] == task]))
        print('Sample number: ', len(merged_pairs_data[merged_pairs_data['task'] == task]['data_id'].unique()))

    merged_pairs_data.to_csv(
        os.path.join('eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct', 'merged_all_pairs_;{}.csv'.format(
            ';'.join(
                [key + '-' + str(dataset_merge_pair_number[key]) + '-' + str(dataset_merge_sample_number[key]) for key
                 in task_list]))))

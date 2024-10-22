import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.metrics import scores
from tqdm import tqdm
from backbone.num_tokens import num_tokens_from_messages
from sup_func.sup_func import printc

custom_dir = "data/datasets"  # not used

download_codeact = False
download_eurus_sft = False
download_eurus_dpo = False
download_ultra_200k = False
download_ultrafeedback = False
download_ultrafeedback_binarized = False
process_ours = True

convert_to_alpaca_format = True
convert_to_tree_data = False  # only for eurus data
load_tree_data = True

select_long_data = False

convert_data_dir = 'TreeDPO/data'
if not os.path.exists(convert_data_dir):
    convert_data_dir = 'TreeDPO/data'


if download_codeact:
    dataset_codeact = load_dataset("xingyaoww/code-act")


    def convert_to_str_dict(turn):
        res = {}
        res['role'] = str(turn['role'])
        res['content'] = str(turn['content'])
        return res


    print('codeact_codeact', len(dataset_codeact.data['codeact']))
    codeact_codeact = []
    for _ in range(10):
        codeact_codeact_item = []
        index = np.random.randint(0, len(dataset_codeact.data['codeact']))
        print(index)
        for item in dataset_codeact.data['codeact']['conversations'][index]:
            item_dict = {}
            if 'role' in item:
                item_dict['role'] = item['role']
            for key in item:
                item_dict[key] = item[key]
            for key in item_dict:
                item_dict[key] = str(item_dict[key])
            print(item_dict)
            codeact_codeact_item.append(item_dict)
        codeact_codeact.append(codeact_codeact_item)

    print('codeact_general', len(dataset_codeact.data['general']))
    codeact_general = []
    for _ in range(10):
        codeact_general_item = []
        index = np.random.randint(0, len(dataset_codeact.data['general']))
        print(index)
        for item in dataset_codeact.data['general']['conversations'][index]:
            item_dict = {}
            if 'role' in item:
                item_dict['role'] = item['role']
            for key in item:
                item_dict[key] = item[key]
            for key in item_dict:
                item_dict[key] = str(item_dict[key])
            print(item_dict)
            codeact_general_item.append(item_dict)
        codeact_general.append(codeact_general_item)

    if convert_to_alpaca_format:
        alpaca_dataset = []
        for ind in tqdm(range(len(dataset_codeact.data['codeact']))):
            new_line = {
                "conversations": [
                    {
                        "from": "human",
                        "value": "user instruction"
                    },
                    {
                        "from": "gpt",
                        "value": "model response"
                    }
                ],
                "system": "system prompt (optional)",
                "tools": "tool description (optional)"
            }

            new_line = {}
            new_line['system'] = ''
            new_line['conversations'] = []
            for turn in dataset_codeact.data['codeact']['conversations'][ind]:
                if str(turn['role']) == 'system':
                    new_line['system'] = str(turn['content'])
                else:
                    new_line['conversations'].append(convert_to_str_dict(turn))

            alpaca_dataset.append(new_line)

        # Write JSON file
        with open(os.path.join(convert_data_dir, 'codeact_codeact.json'), 'w') as f:
            json.dump(alpaca_dataset, f)

        alpaca_dataset = []
        for ind in tqdm(range(len(dataset_codeact.data['general']))):
            new_line = {
                "conversations": [
                    {
                        "from": "human",
                        "value": "user instruction"
                    },
                    {
                        "from": "gpt",
                        "value": "model response"
                    }
                ],
                "system": "system prompt (optional)",
                "tools": "tool description (optional)"
            }

            new_line = {}
            new_line['system'] = ''
            new_line['conversations'] = []
            for turn in dataset_codeact.data['general']['conversations'][ind]:
                if str(turn['role']) == 'system':
                    new_line['system'] = str(turn['content'])
                else:
                    new_line['conversations'].append(convert_to_str_dict(turn))

            alpaca_dataset.append(new_line)

        # Write JSON file
        with open(os.path.join(convert_data_dir, 'codeact_general.json'), 'w') as f:
            json.dump(alpaca_dataset, f)

if download_eurus_sft:
    dataset_eurus_sft = load_dataset("openbmb/UltraInteract_sft")

    print('dataset_eurus_sft', len(dataset_eurus_sft['train']['task']))
    sft_task_count = Counter(dataset_eurus_sft['train']['task'])
    print(sft_task_count)
    sft_dataset_count = Counter(dataset_eurus_sft['train']['dataset'])
    print(sft_dataset_count)

    sft_id_tree_pairs = []
    parent_dict = set(dataset_eurus_sft['train']['parent_id'])
    for id in dataset_eurus_sft['train']['id']:
        if id in parent_dict:
            sft_id_tree_pairs.append(id)

    print(sft_id_tree_pairs)

    print('dataset_eurus_sft', len(dataset_eurus_sft['train']))
    eurus_sft = []
    for _ in range(10):
        index = np.random.randint(0, len(dataset_eurus_sft['train']))
        print(index)
        print(dataset_eurus_sft['train'][index])
        eurus_sft.append(dataset_eurus_sft['train'][index])

    length_list = []
    for item in dataset_eurus_sft['train']['instruction']:
        length_list.append(len(item))
    length_dict = Counter(length_list)

    print(length_dict)

    long_list = []
    for item in dataset_eurus_sft['train']:
        if len(item['instruction']) > 5000:
            long_list.append(item)
    print(json.dumps(long_list[5], indent=2))
    print(json.dumps(long_list[264], indent=2))

    short_list = []
    for item in dataset_eurus_sft['train']:
        if len(item['instruction']) < 2000:
            short_list.append(item)
    print(json.dumps(short_list[1395], indent=2))
    print(json.dumps(short_list[264], indent=2))

    if convert_to_alpaca_format:
        alpaca_dataset = []
        for ind in tqdm(range(len(dataset_eurus_sft['train']))):
            new_line = {
                "instruction": "user instruction (required)",
                "input": "user input (optional)",
                "output": "model response (required)",
                "system": "system prompt (optional)",
                "history": [
                    ["user instruction in the first round (optional)", "model response in the first round (optional)"],
                    ["user instruction in the second round (optional)", "model response in the second round (optional)"]
                ]
            }
            new_line = {}
            line = dataset_eurus_sft['train'][ind]
            new_line['instruction'] = line['instruction']
            new_line['output'] = line['response']

            new_line['input'] = ''

            alpaca_dataset.append(new_line)

        # Write JSON file
        with open(os.path.join(convert_data_dir, 'eurus_sft.json'), 'w') as f:
            json.dump(alpaca_dataset, f)

if download_eurus_dpo:
    dataset_eurus_dpo = load_dataset("openbmb/UltraInteract_pair")

    print('dataset_eurus_dpo')
    dpo_task_count = Counter(dataset_eurus_dpo['train']['task'])
    print(dpo_task_count)
    dpo_dataset_count = Counter(dataset_eurus_dpo['train']['dataset'])
    print(dpo_dataset_count)

    dpo_id_tree_pairs = []
    parent_dict = set(dataset_eurus_dpo['train']['parent_id'])
    for id in dataset_eurus_dpo['train']['id']:
        if id in parent_dict:
            dpo_id_tree_pairs.append(id)

    print(dpo_id_tree_pairs)

    print('dataset_eurus_dpo')
    eurus_dpo = []
    for _ in range(10):
        index = np.random.randint(0, len(dataset_eurus_dpo['train']))
        print(index)
        print(dataset_eurus_dpo['train'][index])
        eurus_dpo.append(dataset_eurus_dpo['train'][index])

    length_list = []
    for item in dataset_eurus_dpo['train']['trajectory']:
        length_list.append(len(item))
    length_dict = Counter(length_list)

    print(length_dict)

    more_than_one_turn_data = []
    single_turn_data = []
    for item in dataset_eurus_dpo['train']['trajectory']:
        if len(item) > 1:
            more_than_one_turn_data.append([str(step) for step in item])
        else:
            single_turn_data.append([str(step) for step in item])

    if convert_to_tree_data:

        # filter the multi-turn instructions, and unique them
        single_turn_data_str = [str(item) for item in single_turn_data]
        single_turn_data_str_count = Counter(single_turn_data_str)
        single_turn_data_str_keys = list(single_turn_data_str_count.keys())
        single_turn_data_str_keys = [eval(item) for item in single_turn_data_str_keys]

        more_than_one_turn_data_str = [str(item) for item in more_than_one_turn_data]
        more_than_one_turn_data_str_count = Counter(more_than_one_turn_data_str)
        more_than_one_turn_data_str_keys = list(more_than_one_turn_data_str_count.keys())
        more_than_one_turn_data_str_keys = [eval(item) for item in more_than_one_turn_data_str_keys]

        for _ in range(10):
            index = np.random.randint(0, len(more_than_one_turn_data))
            print(index)
            print(more_than_one_turn_data[index])
            eurus_dpo.append(more_than_one_turn_data[index])

        # check the relation between single turn and multi turn ones

        single_tree_pair = []

        for item_multi in more_than_one_turn_data:
            if str([item_multi[0]]) in single_turn_data_str_count:
                single_tree_pair.append(item_multi)

        # Get the N-gram data for multi-turn data
        more_than_one_turn_data_str_keys_grams = []
        for item in single_turn_data_str_keys + more_than_one_turn_data_str_keys:
            for ind in range(len(item)):
                more_than_one_turn_data_str_keys_grams.append(item[0:(ind + 1)])

        more_than_one_turn_data_str_keys_grams_str = [str(item) for item in more_than_one_turn_data_str_keys_grams]
        more_than_one_turn_data_str_keys_grams_count = Counter(more_than_one_turn_data_str_keys_grams_str)

        # sort the multi-turn data, preparing for get the tree data structure
        sorted_grams = list(more_than_one_turn_data_str_keys_grams_count.keys())
        sorted_grams.sort(reverse=False)

        more_than_one_turn_data_str_keys_grams_count_pd = []
        for key in sorted_grams:
            item = eval(key)
            more_than_one_turn_data_str_keys_grams_count_pd.append(
                [key, len(item), more_than_one_turn_data_str_keys_grams_count[key]])

        more_than_one_turn_data_str_keys_grams_count_pd = pd.DataFrame(more_than_one_turn_data_str_keys_grams_count_pd,
                                                                       columns=['item', 'length', 'count'])

        # Contruct the tree structured data
        # assert the data in more_than_one_turn_data_str_keys_grams_count_pd is sorted according to the dfs order due to the sorting in Counter:
        tree_organized_data = []

        currect_tree = None
        dpo_count = 0
        current_length = 1

        repeat_branch = []
        repeat_branch_tree = []
        tree_id = 0
        tree_pd_id_map = {}
        for ind, item in tqdm(more_than_one_turn_data_str_keys_grams_count_pd.iterrows()):
            if item['length'] >= current_length and current_length != 1:
                repeat_branch.append(ind)
                repeat_branch_tree.append(tree_id - 1)
                dpo_count = 0
                current_length = item['length']

            if item['length'] >= current_length:
                if currect_tree is not None:
                    tree_organized_data.append(currect_tree)
                currect_tree = []
                dpo_count = 0
                tree_pd_id_map[tree_id] = ind
                tree_id += 1

            if item['count'] > dpo_count:
                currect_tree.append(eval(item['item']))
                dpo_count = item['count']

            current_length = item['length']
        tree_organized_data.append(currect_tree)
        tree_pd_id_map[tree_id] = ind

        all_data_set = set(single_turn_data_str_count.keys()) | set(more_than_one_turn_data_str_count.keys())
        select_view = []
        for ind in repeat_branch_tree:
            current_tree_new = []
            for item in tree_organized_data[ind]:
                if str(item) in all_data_set:
                    current_tree_new.append(item)
            select_view.append([tree_organized_data[ind], current_tree_new])
            tree_organized_data[ind] = current_tree_new

        # reconstruct the tree to be sequential data, labeled with the tree label

        tree_id_map = {}
        for ind, tree in tqdm(enumerate(tree_organized_data)):
            for branch in tree[::-1]:
                assert str(branch) not in tree_id_map
                tree_id_map[str(branch)] = ind

        dpo_dataset_full_pd = pd.DataFrame(dataset_eurus_dpo['train'])


        def map_to_str(trajectory):
            trajectory_new = []
            for item in trajectory:
                trajectory_new.append(str(item))
            return str(trajectory_new)


        for item in dpo_dataset_full_pd['trajectory']:
            assert map_to_str(item) in tree_id_map

        dpo_dataset_full_pd['tree_id'] = dpo_dataset_full_pd['trajectory'].apply(lambda x: tree_id_map[map_to_str(x)])
        dpo_dataset_full_pd['depth'] = dpo_dataset_full_pd['trajectory'].apply(lambda x: len(x))

        dpo_dataset_full_pd_sort = dpo_dataset_full_pd.sort_values(by=['tree_id', 'depth'])

        dpo_dataset_full_pd_sort.to_csv(os.path.join('data', 'tree_structured_preference.csv'))

    if load_tree_data:
        dpo_dataset_full_pd_sort = pd.read_csv(os.path.join('data', 'tree_structured_preference.csv'))
        sample = dpo_dataset_full_pd_sort.loc[dpo_dataset_full_pd_sort['tree_id'] == 9][
            ['trajectory', 'chosen', 'rejected']].to_dict('records')
        for item in sample:
            item['trajectory'] = eval(item['trajectory'])
        print(json.dumps(sample, indent=2))
        print(1)

    if convert_to_alpaca_format:
        alpaca_dataset = []
        for ind in tqdm(range(len(dataset_eurus_dpo['train']))):
            new_line = {
                "instruction": "user instruction (required)",
                "input": "user input (optional)",
                "chosen": "chosen answer (required)",
                "rejected": "rejected answer (required)",
                "system": "system prompt (optional)",
                "history": [
                    ["user instruction in the first round (optional)", "model response in the first round (optional)"],
                    ["user instruction in the second round (optional)", "model response in the second round (optional)"]
                ]
            }
            new_line = {}
            new_line['history'] = []

            line = dataset_eurus_dpo['train'][ind]
            whole_history = line['trajectory']
            assert whole_history[-1]['from'] == 'user'
            history = whole_history[:-1]
            assert len(history) % 2 == 0

            new_line['instruction'] = whole_history[-1]['value']
            for ind in range(0, len(history), 2):
                assert history[ind]['from'] == 'user'
                assert history[ind + 1]['from'] == 'assistant'
                new_line['history'].append([history[ind]['value'], history[ind + 1]['value']])
            new_line['chosen'] = line['chosen']
            new_line['rejected'] = line['rejected']

            new_line['input'] = ''

            alpaca_dataset.append(new_line)

        # Write JSON file
        with open(os.path.join(convert_data_dir, 'eurus_dpo.json'), 'w') as f:
            json.dump(alpaca_dataset, f)

if download_ultra_200k:
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k")

if download_ultrafeedback:
    dataset = load_dataset('openbmb/UltraFeedback')

    if convert_to_alpaca_format:
        alpaca_dataset = []
        for ind in tqdm(range(len(dataset['train']))):

            line = dataset['train'][ind]
            scores = []
            for sample in line['completions']:
                scores.append(sample['overall_score'])
            if len(scores) > 1:
                sort_index = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)
                for position, chosen_ind in enumerate(sort_index[:-1]):
                    for reject_ind in sort_index[(position + 1):]:
                        chosen_sample = line['completions'][chosen_ind]
                        reject_sample = line['completions'][reject_ind]
                        if chosen_sample['overall_score'] > reject_sample['overall_score']:
                            new_line = {
                                "instruction": "user instruction (required)",
                                "input": "user input (optional)",
                                "chosen": "chosen answer (required)",
                                "rejected": "rejected answer (required)",
                                "system": "system prompt (optional)",
                                "history": [
                                    ["user instruction in the first round (optional)",
                                     "model response in the first round (optional)"],
                                    ["user instruction in the second round (optional)",
                                     "model response in the second round (optional)"]
                                ]
                            }
                            new_line = {}
                            new_line['history'] = []
                            new_line['instruction'] = line['instruction']
                            new_line['chosen'] = chosen_sample['response']
                            new_line['rejected'] = reject_sample['response']

                            new_line['input'] = ''

                            alpaca_dataset.append(new_line)

        print(len(alpaca_dataset))
        # Write JSON file
        with open(os.path.join(convert_data_dir, 'ultrafeedback_allpairs.json'), 'w') as f:
            json.dump(alpaca_dataset, f)

if download_ultrafeedback_binarized:

    dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    if convert_to_alpaca_format:
        alpaca_dataset = []
        for ind in tqdm(range(len(dataset['train']))):
            new_line = {
                "instruction": "user instruction (required)",
                "input": "user input (optional)",
                "chosen": "chosen answer (required)",
                "rejected": "rejected answer (required)",
                "system": "system prompt (optional)",
                "history": [
                    ["user instruction in the first round (optional)", "model response in the first round (optional)"],
                    ["user instruction in the second round (optional)", "model response in the second round (optional)"]
                ]
            }
            new_line = {}
            new_line['history'] = []

            line = dataset_eurus_dpo['train'][ind]
            whole_history = line['trajectory']
            assert whole_history[-1]['from'] == 'user'
            history = whole_history[:-1]
            assert len(history) % 2 == 0

            new_line['instruction'] = whole_history[-1]['value']
            for ind in range(0, len(history), 2):
                assert history[ind]['from'] == 'user'
                assert history[ind + 1]['from'] == 'assistant'
                new_line['history'].append([history[ind]['value'], history[ind + 1]['value']])
            new_line['chosen'] = line['chosen']
            new_line['rejected'] = line['rejected']

            new_line['input'] = ''

            alpaca_dataset.append(new_line)

        # Write JSON file
        with open(os.path.join(convert_data_dir, 'ultrafeedback_binarized.json'), 'w') as f:
            json.dump(alpaca_dataset, f)

if process_ours:
    dataset = pd.read_csv(
        # 'eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct/merged_all_pairs_;math-22331-694.csv',
        # 'eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct/merged_all_pairs_;math-22331-1388.csv',
        'eval_results/codeact_agent_tree.Meta-Llama-3-8B-Instruct/merged_all_pairs_;math-22331-2334.csv',
        # nrows=2000
    )
    dataset['positive_history'] = dataset['positive_history'].apply(lambda x: eval(x))
    dataset['negative_history'] = dataset['negative_history'].apply(lambda x: eval(x))
    dataset['positive_run_index'] = dataset['positive_run_index'].apply(lambda x: eval(x))
    dataset['negative_run_index'] = dataset['negative_run_index'].apply(lambda x: eval(x))
    if convert_to_alpaca_format:
        alpaca_dataset = []
        length = []
        task_count = []
        for ind, line in tqdm(dataset.iterrows()):
            new_line = {
                "instruction": "user instruction (required)",
                "input": "user input (optional)",
                "chosen": "chosen answer (required)",
                "rejected": "rejected answer (required)",
                "system": "system prompt (optional)",
                "history": [
                    ["user instruction in the first round (optional)", "model response in the first round (optional)"],
                    ["user instruction in the second round (optional)", "model response in the second round (optional)"]
                ]
            }
            new_line = {}
            new_line['history'] = []
            pos_len = num_tokens_from_messages(line['positive_history'][0:(2 + 2 * line['contrast_step'])],
                                               'Meta-Llama-3-8B-Instruct')
            neg_len = num_tokens_from_messages(line['negative_history'][0:(2 + 2 * line['contrast_step'])],
                                               'Meta-Llama-3-8B-Instruct')
            length.append(pos_len)
            length.append(neg_len)
            # if not select_long_data or (
            #         pos_len >= 7416.1 and pos_len <= 8196 or (neg_len >= 7416.1 and neg_len <= 8196)):
            if not select_long_data and (pos_len <= 4096 and neg_len <= 4096):

                whole_history = line['positive_history'][0:(1 + 2 * line['contrast_step'])]
                assert whole_history == line['negative_history'][0:(1 + 2 * line['contrast_step'])]
                assert whole_history[-1]['role'] == 'user'
                history = whole_history[:-1]
                assert len(history) % 2 == 0

                new_line['instruction'] = whole_history[-1]['content']
                for ind in range(0, len(history), 2):
                    assert history[ind]['role'] == 'user'
                    assert history[ind + 1]['role'] == 'assistant'
                    new_line['history'].append([history[ind]['content'], history[ind + 1]['content']])

                assert line['positive_history'][1 + 2 * line['contrast_step']]['content'] != \
                       line['negative_history'][1 + 2 * line['contrast_step']]['content']
                new_line['chosen'] = line['positive_history'][1 + 2 * line['contrast_step']]['content']
                new_line['rejected'] = line['negative_history'][1 + 2 * line['contrast_step']]['content']
                # new_line['output'] = [line['positive_history'][1+2*line['contrast_step']]['content'], line['negative_history'][1+2*line['contrast_step']]['content']]

                new_line['input'] = ''

                alpaca_dataset.append(new_line)

                task_count.append(line['task'])
        # if select_long_data:
        #     alpaca_dataset_new = []
        #     for i in range(100):
        #         alpaca_dataset_new += alpaca_dataset
        #     alpaca_dataset = alpaca_dataset_new
        for line in alpaca_dataset:
            for turn in line['history']:
                printc(turn[0], 'yellow')
                printc(turn[1], 'blue')
            printc(line['instruction'], 'yellow')
            printc(line['chosen'], 'green')
            printc(line['rejected'], 'red')

        print('Original Pair Number: ', len(dataset))
        print('Filtered Pair Number', len(alpaca_dataset))

        print('Task Count', Counter(task_count))

        hist, bin_edges = np.histogram(length, bins=10)
        print(bin_edges)
        print(hist)
        # Write JSON file
        with open(os.path.join(convert_data_dir,
                               # 'Tree_DPO_temperature_math-22331-694.json'
                               # 'Tree_DPO_temperature_math-22331-1388.json'
                               'Tree_DPO_temperature_math-22331-2334.json'
                               ),
                  'w') as f:
            json.dump(alpaca_dataset, f)

    print(1)

print('Finished')

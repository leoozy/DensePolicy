import copy
import pdb
import random

import math
import numpy as np
import pandas as pd

from model.codeact_agent.codeact_agent import codeact_agent
from sup_func.sup_func import printc

meta_policy_pn = '''\n\nBefore the individual starts the aforementioned task step, kindly supply them with overarching guidelines for this task step. The guidelines must focus on the general approach for this stage, without mentioning specific actions tied to the task. Please provide one positive and one negative principle, formatted as a dictionary:

{ 'Positive': # Your positive principle, 'Negative': # Your negative principle }'''

meta_policy_op = '''"\n\nBefore the individual starts the aforementioned task, kindly supply them with overarching principle for this task. The principle must focus on the general thought for this stage, without mentioning specific actions tied to the task. Please provide one principle that you consider most advantageous and another that may appear suitable but is not the most effective, formatted as a dictionary:

{ 'Optimal': # Your optimal principle, 'Suboptimal': # Your suboptimal principle }"'''

policy_direct = {1: '''In this step, for practice purposes, please create an incorrect step that exemplifies a common policy mistake in this step. 
Avoid explaining the reason behind its incorrectness or how you make the mistake, and proceed as if you were carrying out the task regularly. 
Ensure not to replicate any prior error steps.''', 2: ''}

policy_direct_2 = {1: '''Although you have an original plan for what to do next, we request that you intentionally create something less efficient than what to do next. You can still respond using either 'Action:' or 'Answer:' depends on your original plan for what to do next.
Avoid explaining the reasons for its inefficiency in your thoughts or within the code comments, and proceed as if you were carrying out this step regularly. 
Ensure not to replicate prior steps.''', 2: ''}

policy_direct_3 = {1: '''Now, we request that you intentionally produce a less efficient output than this one:
```
{}
```
However, if there is no less efficient output, you may output the same content, such as the output with "Answer:"..
Avoid explaining the reasons for its inefficiency in your thoughts or within the code comments.
''', 2: ''}

policy_direct_4 = {1: '''For this step, we ask you to modify the correct step shown below into an incorrect step that has planning mistakes:
```
{}
```
Avoid explaining the reasons for its worse plan in your thoughts or within the code comments.
''', 2: '''For this step, we ask you to intentionally produce this turn's output with better plan than the one shown below:
```
{}
```
Avoid explaining the reasons for its better plan in your thoughts or within the code comments.
'''}

# for practice purposes to achieve suboptimal results
# We want you to intentionally perform less effectively in this step comparing to what you original will do at this step.

policy_reflection = {'1': '', '2': ''}


def add_policy(messages, policy, position):
    res_message = copy.deepcopy(messages)
    if position == 'end':
        assert res_message[-1]['role'] == 'user'
        # Remove newline characters from the end
        s_stripped = res_message[-1]['content'].rstrip('\n')
        num_newlines = len(res_message[-1]['content']) - len(s_stripped)
        # Create a string with the removed newline characters
        newlines = '\n' * num_newlines
        res_message[-1]['content'] = s_stripped + '\n\n' + policy + newlines

    elif position == 'sys':
        res_message = [{"role": "system", "content": policy}] + res_message
    else:
        raise ValueError('position {} is not supported'.format(position))
    return res_message


def generate_step_level_policy(backbone_func, label, messages, last_normal_generation='',
                               access_to_normal_generation=False, step_level=False):
    if label == 0:
        return ''
    else:
        if not step_level:
            if not access_to_normal_generation:
                return policy_direct_2[label]
            else:
                action_type, action, extra_info = codeact_agent.action_parser(last_normal_generation)
                if action_type == 'answer':
                    return ''
                else:
                    return policy_direct_4[label].format(last_normal_generation)
        else:
            message_for_policy = add_policy(messages, meta_policy_pn, 'end')
            policy, _ = backbone_func(message_for_policy)
            policy_dict = eval(policy)
            # return policy_dict[{1: 'Suboptimal', 2: 'Optimal'}[label]]

            return policy_dict[{1: 'Negative', 2: 'Positive'}[label]]


def get_policy_temperature(label):
    if label == 0:
        return 0.0
    elif label % 3 == 1:
        return 1.0
    elif label % 3 == 2:
        return 0.5
    else:
        return 0.7
        # raise ValueError('label {} is not supported'.format(label))


def generate_tree_leaf_ids(n, base):
    sequence = []
    for length in range(1, n + 1):
        for i in range(base ** length):
            base_repr = ''
            num = i
            while num:
                base_repr = str(num % base) + base_repr
                num //= base
            base_repr = '0' * (length - len(base_repr)) + base_repr  # Pad with zeros to the left
            sequence.append(
                tuple([int(digit) for digit in base_repr]))  # Convert the representation to a list of integers
    return sequence


def restore_task(env, messages):
    last_example_start = 0
    for last_example_start in range(len(messages) - 1, -1, -1):
        if messages[last_example_start]['role'] == 'user' and "Now, let's move on to the next task." in \
                messages[last_example_start]['content']:
            break
    print('Last example index: {}'.format(last_example_start))

    env.reset()
    for ind, message in enumerate(messages[last_example_start:]):
        if message["role"] == "assistant":
            if 'TimeoutError' not in messages[last_example_start + ind + 1]['content']:
                action_type, action, extra_info = codeact_agent.action_parser(message['content'])
                env, state, done = codeact_agent.execute_action(
                    env, action_type, action, extra_info
                )
                if state != messages[last_example_start + ind + 1]['content']:
                    print('Warning! Mismatch restored state')
                    with open('restore_task_mismatch_case.txt', 'a') as f:
                        f.write(
                            '{}\n\n{}\n\n{}\n\n\n'.format(action, state,
                                                          messages[last_example_start + ind + 1]['content']))
            else:
                print('Time out step skipped')
    return env


class NBase_Tree:
    def __init__(self, base, init_value):
        self.base = base
        self.init_value = copy.deepcopy(init_value)
        self.reset_values()

    def reset_values(self, new_init_value=None):
        if new_init_value is not None:
            self.values = copy.deepcopy(new_init_value)
        else:
            self.values = {}
        # for key in generate_tree_leaf_ids(self.depth, self.base):
        #     self.values[key] = init_value

    def keys(self):
        return self.values.keys()

    def __getitem__(self, index):
        if index not in self.values:
            self.values[index] = copy.deepcopy(self.init_value)
        return self.values[index]

    def __setitem__(self, key, value):
        self.values[key] = value

    def is_root(self, index):
        return len(index) == 1

    def parent_node_value(self, index):
        if self.is_root(index):
            raise ValueError('No parent node for the root')
        else:
            return self[index[0:-1]]

    def child_nodes(self, index):
        res = []
        for ind in range(self.base):
            res.append(index + (ind,))
        return res

    def get_child_values(self, index, key):
        child_values = []
        for child_index in self.child_nodes(index):
            if child_index in self.values:
                child_values.append(self[child_index][key])
        return child_values

    def same_parent_node(self, index):
        res = []
        for ind in range(0, self.base):
            if ind != index[-1]:
                res.append(index[:-1] + (ind,))
        return res

    def same_parent_node_for_traversal(self, index):
        res = []
        for ind in range(index[-1] + 1, self.base):
            res.append(index[:-1] + (ind,))
        return res

    def retrieve_path(self, index, key):
        res = []
        for i in range(len(index)):
            res.append(self.values[index[:(i + 1)]][key])
        return res


class Contrast_Pair_MCTree(NBase_Tree):
    def __init__(self, base, max_depth):
        super().__init__(base, {})
        self.depth = max_depth
        self.contrast_pair_num = 0
        self.node_MCTree = NBase_Tree(base, {})

    @staticmethod
    def is_contrast_index(index_1, index_2):
        diff = False
        for contrast_step in range(min(len(index_1), len(index_2))):
            if index_1[contrast_step] != index_2[contrast_step]:
                diff = True
                break
        flag = diff and (all(x == 0 for x in index_1[(contrast_step + 1):]) and all(
            x == 0 for x in index_2[(contrast_step + 1):]))
        return flag, contrast_step

    @staticmethod
    def find_reverse_non_zero_index(lst):
        for i in range(len(lst) - 1, -1, -1):
            if lst[i] != 0:
                return i
        return -1

    def get_contrast_pair_num(self):
        res_check = 0
        for key in self.values.keys():
            res_check += self[key]["num_contrast_pairs"]
        # assert self.contrast_pair_num == res_check
        assert self.contrast_pair_num * 2 == res_check
        return self.contrast_pair_num

    def update_node_score(self, index, res, history=None):
        for node_step in range(len(index)):
            node_index = index[:(1 + node_step)]
            if node_index not in self.node_MCTree.values:
                self.node_MCTree[node_index] = {
                    "num_node_explore": 1,
                    "node_score_sum": res,
                    "node_score": res,
                    "node_score_each": [res],
                    "node_score_depth": [len(index) - (node_step + 1)],
                    "node_score_depth_avg": len(index) - (node_step + 1)
                }
                if history is not None:
                    self.node_MCTree[node_index]["history"] = history[:(2 + 2 * node_step)]
            else:
                self.node_MCTree[node_index]["num_node_explore"] += 1
                self.node_MCTree[node_index]['node_score_sum'] += res
                self.node_MCTree[node_index]['node_score'] = self.node_MCTree[node_index]['node_score_sum'] / \
                                                             self.node_MCTree[node_index]['num_node_explore']
                self.node_MCTree[node_index]['node_score_each'].append(res)
                self.node_MCTree[node_index]['node_score_depth'].append(len(index) - (node_step + 1))
                self.node_MCTree[node_index]['node_score_depth_avg'] = np.mean(
                    self.node_MCTree[node_index]['node_score_depth'])

                if history is not None:
                    assert self.node_MCTree[node_index]["history"] == history[:(2 + 2 * node_step)]

    def update_score(self, index, res, no_info=False):
        assert index not in self.values
        self[index] = {
            "num_pairs": 0,
            "num_contrast_pairs": 0,
            "score": res,
            "num_explore": 0,
            "node_depth": self.find_reverse_non_zero_index(index)
        }
        if not no_info:
            printc('Updating {}'.format(index), 'cyan')
        for key in self.values.keys():
            flag, contrast_step = self.is_contrast_index(index, key)
            if flag:
                # # update index MC record
                # if self[key]['node_depth'] <= self[index]['node_depth']:
                #     self[key]["num_pairs"] += 1
                #     if res != self[key]["score"]:
                #         self[key]["num_contrast_pairs"] += 1
                #         if not no_info:
                #             printc('{} contrast pair of {}'.format(index, key), 'magenta')
                #         self.contrast_pair_num += 1
                #     else:
                #         if not no_info:
                #             printc('{} pair of {}'.format(index, key), 'cyan')
                # else:
                #     self[index]["num_pairs"] += 1
                #     if res != self[key]["score"]:
                #         self[index]["num_contrast_pairs"] += 1
                #         if not no_info:
                #             printc('{} contrast pair of {}'.format(key, index), 'magenta')
                #         self.contrast_pair_num += 1
                #     else:
                #         if not no_info:
                #             printc('{} pair of {}'.format(key, index), 'cyan')
                # update index MC record
                self[key]["num_pairs"] += 1
                self[index]["num_pairs"] += 1
                if res != self[key]["score"]:
                    self[key]["num_contrast_pairs"] += 1
                    self[index]["num_contrast_pairs"] += 1
                    if not no_info:
                        printc('{} contrast pair of {}'.format(index, key), 'magenta')
                    self.contrast_pair_num += 1
                else:
                    if not no_info:
                        printc('{} pair of {}'.format(index, key), 'cyan')

    def update_explore(self, index):
        assert index in self.values
        self[index]["num_explore"] += 1

    def max_MC_score_item(self):
        mean_score = np.mean([self[key]['score'] for key in self.values.keys()])
        printc('Average Score: {}'.format(str(mean_score)), 'cyan')
        all_scores = {}
        all_score_show = {}
        for index in self.values:
            score_performance = np.abs(self[index]["score"] - 1 / (1 + np.exp(-64 * (mean_score - 0.5))))
            score_contrast_ratio = np.power(self[index]["num_contrast_pairs"] / (self[index]['num_pairs'] + 1e-8),
                                            1 / 4)
            score_depth = ((self.depth - self[index]['node_depth']) / self.depth)
            score_explore = 2 * np.sqrt(1 + self[index]["num_explore"]) / (2 + self[index]["num_explore"])
            # score_explore = np.power(0.8, self[index]["num_explore"])

            # all_scores[index] = 0.3 * score_performance + 0.3 * score_contrast_ratio + 0.4 * score_depth + score_explore
            all_scores[index] = 0.5 * score_contrast_ratio + 0.5 * score_depth + score_explore
            all_score_show[index] = [round(score_performance, 2), round(score_contrast_ratio, 2), round(score_depth, 2),
                                     round(score_explore, 2), round(all_scores[index], 2)]
            # 0.125^(1-abs(score-mean)) is 0.125 for 0, about 0.35 for 0.5, and 1 for 1; difference between score 0 and 1 is nearly abs(mean_score-0.5); with 0.5 coefficient
            # np.abs(self[index]["score"] - 1 / (1 + np.exp(-64 * (mean_score - 0.5)))) is nearly 1 for fewer score
            # (num_contrast_pairs/num_pairs)^(1/4) is about 0.55 for 0.1, and 0.75 for 0.3; with 0.5 coefficient
            # 0.8^num_explore change from 1 to 0.1 when num_explore change from 0 to 10: explore one node for 10 times
            # 2 * np.sqrt(1 + self[index]["num_explore"]) / (2 + self[index]["num_explore"]) is about 0.95 for 1, 0.7 for 5, 0.5 for 13 and 0.1 for 396

        printc(str(all_score_show), 'cyan')

        max_value = max(all_scores.values())
        max_index = [key for key, value in all_scores.items() if value == max_value]
        chosen = random.choice(max_index)
        self.update_explore(chosen)
        printc('Chosen {}'.format(chosen), 'background_cyan')
        return chosen

    def propose_explore(self, sequence, max_steps, explored_result):
        res = []

        # sequence = list(copy.deepcopy(explored_result['run_index'].iloc[-1]))

        last_one_index = self.find_reverse_non_zero_index(sequence)
        start = last_one_index + 1
        # end = min(len(sequence), start + self.window_size)
        end = len(sequence)
        printc(
            "Sequence {} Start {} End {}".format(sequence, start, end),
            'magenta')
        for new_policy_index in range(1, self.base):
            for i in range(start, end):
                sequence_new = copy.deepcopy(list(sequence))
                assert sequence_new[i] == 0
                sequence_new[i] = new_policy_index
                # expand to max length:
                sequence_new = sequence_new + [0] * (max_steps - len(sequence_new))
                assert len(sequence_new) == max_steps
                index_new = tuple(sequence_new)

                printc("New Sequence {}".format(index_new), 'magenta')

                # check for index sequences that have been run
                def check_subsequence(row, proposal):
                    return proposal[0:len(row)] == row

                run_detect_res = explored_result['run_index'].apply(check_subsequence, proposal=index_new)
                if not run_detect_res.any():
                    res.append(index_new)
                else:
                    subsequence_df = explored_result.loc[run_detect_res]
                    # Print the 'run_index' values that have the subsequence
                    printc('The following run_index is contained in the new sequence:', 'red')
                    for index, row in subsequence_df.iterrows():
                        printc(str(row['run_index']), 'red')
        return res

    def next_policy_index_generator(self, explored_result, update_last_k, max_steps, window_size):
        if len(explored_result) == 0:
            return [tuple([0] * (max_steps))]
        else:
            # ToDo Make different policy for failed case
            printc("Current Average Performance: {}".format(explored_result['res_cur'].mean()), 'magenta')

            for _, line in explored_result.tail(update_last_k).iterrows():
                self.update_score(line['run_index'], line['res_cur'])

            sequence = self.max_MC_score_item()

            proposed = self.propose_explore(sequence, max_steps, explored_result)

            if window_size > 0:
                chosen = random.sample(proposed, min(len(proposed), window_size))
            else:
                chosen = proposed
            printc('New Iteration Over {}'.format(chosen), 'background_magenta')
            return chosen


class MCValue_MCTree(NBase_Tree):
    def __init__(self, base, max_depth):
        super().__init__(base, {})
        self.depth = max_depth
        self.contrast_pair_num = 0
        # self.node_MCTree = NBase_Tree(base, {})

    # @staticmethod
    # def is_contrast_index(index_1, index_2):
    #     diff = False
    #     for contrast_step in range(min(len(index_1), len(index_2))):
    #         if index_1[contrast_step] != index_2[contrast_step]:
    #             diff = True
    #             break
    #     flag = diff and (all(x == 0 for x in index_1[(contrast_step + 1):]) and all(
    #         x == 0 for x in index_2[(contrast_step + 1):]))
    #     return flag, contrast_step

    # @staticmethod
    # def find_reverse_non_zero_index(lst):
    #     for i in range(len(lst) - 1, -1, -1):
    #         if lst[i] != 0:
    #             return i
    #     return -1

    def get_contrast_pair_num(self):
        contrast_pair_num = 0
        for index_1 in self.values:
            for index_2 in self.same_parent_node_for_traversal(index_1):
                if index_2 in self.values:
                    if abs(self[index_1]['score'] - self[index_2][
                        'score']) >= 0.5:
                        thre_node1 = min(math.floor(np.power(2, self[index_1]['score_depth_avg'])), 8)
                        thre_node2 = min(math.floor(np.power(2, self[index_2]['score_depth_avg'])), 8)
                        if self[index_1]['num_explore'] >= thre_node1 and self[index_2]['num_explore'] >= thre_node2:
                            contrast_pair_num += 1
        self.contrast_pair_num = contrast_pair_num
        return contrast_pair_num

    def get_contrast_pairs(self, filtered=False):
        res = []
        for index_1 in self.values:
            for index_2 in self.same_parent_node_for_traversal(index_1):
                if index_2 in self.values:
                    thre_node1 = min(math.floor(np.power(2, self[index_1]['score_depth_avg'])), 8)
                    thre_node2 = min(math.floor(np.power(2, self[index_2]['score_depth_avg'])), 8)
                    score_diff = abs(self[index_1]['score'] - self[index_2]['score'])
                    assert score_diff <= np.max(self[index_1]['contrast_with_pairs'])
                    assert score_diff <= np.max(self[index_2]['contrast_with_pairs'])
                    res.append(
                        [index_1, self[index_1]['score'], self[index_1]['num_explore'],
                         self[index_1]['score_depth_avg'], thre_node1, index_2,
                         self[index_2][
                             'score'], self[index_2]['num_explore'], self[index_2]['score_depth_avg'], thre_node2,
                         score_diff
                         ])
        res = pd.DataFrame(res, columns=['index_1', 'score_1', 'num_explore_1', 'depth_1', 'thre_node_1', 'index_2',
                                         'score_2', 'num_explore_2', 'depth_2', 'thre_node_2', 'score_diff'])
        res['pass_filter'] = (abs(res['score_1'] - res['score_2']) >= 0.5) & (
                res['num_explore_1'] >= res['thre_node_1']) & (
                                     res['num_explore_2'] >= res['thre_node_2'])
        return res

    def check_scores(self):
        for index_1 in self.values:
            child_values = self.get_child_values(index_1, 'score')
            if len(child_values) > 0:
                assert self[index_1]['score'] == np.mean(child_values)

    def update_node_score(self, index, res, history=None, no_info=False):
        for node_step in range(len(index) - 1, -1, -1):
            node_index = index[:(1 + node_step)]
            if node_index not in self.values:
                assert self.get_child_values(index, 'score') == []
                self[node_index] = {
                    "num_explore": 1,
                    "score_sum": res,
                    "score": res,
                    "score_each": [res],
                    "score_depth": [len(index) - (node_step + 1)],
                    "score_depth_avg": len(index) - (node_step + 1),
                    "contrast_with_pairs": [0] * self.base
                    # "num_pairs": 0,
                    # "num_contrast_pairs": 0,
                }
                if history is not None:
                    self[node_index]["history"] = history[:(2 + 2 * node_step)]
            else:
                self[node_index]["num_explore"] += 1
                self[node_index]['score_sum'] += res
                # self[node_index]['score'] = self[node_index]['score_sum'] / \
                #                             self[node_index]['num_explore']
                child_values = self.get_child_values(node_index, 'score')
                assert len(child_values) > 0
                self[node_index]['score'] = np.mean(child_values)
                self[node_index]['score_each'].append(res)
                self[node_index]['score_depth'].append(len(index) - (node_step + 1))
                self[node_index]['score_depth_avg'] = np.mean(
                    self[node_index]['score_depth'])
                self[node_index]['num_explore'] += 1

                if history is not None:
                    assert self[node_index]["history"] == history[:(2 + 2 * node_step)]
            for index_leaf in self.same_parent_node(node_index):
                if index_leaf in self.values:
                    contrast = abs(self[node_index]['score'] - self[index_leaf]['score'])
                    self[node_index]['contrast_with_pairs'][index_leaf[-1]] = contrast
                    self[index_leaf]['contrast_with_pairs'][node_index[-1]] = contrast
                    if not no_info:
                        printc('{} pair of {}, contrast {}'.format(node_index, index_leaf, contrast), 'cyan')

    # def update_score(self, index, res, no_info=False):
    #     assert index not in self.values
    #     self[index] = {
    #         "num_pairs": 0,
    #         "num_contrast_pairs": 0,
    #         "score": res,
    #         "num_explore": 0,
    #         "node_depth": self.find_reverse_non_zero_index(index)
    #     }
    #     if not no_info:
    #         printc('Updating {}'.format(index), 'cyan')
    #     for key in self.values.keys():
    #         flag, contrast_step = self.is_contrast_index(index, key)
    #         if flag:
    #             # # update index MC record
    #             # if self[key]['node_depth'] <= self[index]['node_depth']:
    #             #     self[key]["num_pairs"] += 1
    #             #     if res != self[key]["score"]:
    #             #         self[key]["num_contrast_pairs"] += 1
    #             #         if not no_info:
    #             #             printc('{} contrast pair of {}'.format(index, key), 'magenta')
    #             #         self.contrast_pair_num += 1
    #             #     else:
    #             #         if not no_info:
    #             #             printc('{} pair of {}'.format(index, key), 'cyan')
    #             # else:
    #             #     self[index]["num_pairs"] += 1
    #             #     if res != self[key]["score"]:
    #             #         self[index]["num_contrast_pairs"] += 1
    #             #         if not no_info:
    #             #             printc('{} contrast pair of {}'.format(key, index), 'magenta')
    #             #         self.contrast_pair_num += 1
    #             #     else:
    #             #         if not no_info:
    #             #             printc('{} pair of {}'.format(key, index), 'cyan')
    #             # update index MC record
    #             self[key]["num_pairs"] += 1
    #             self[index]["num_pairs"] += 1
    #             if res != self[key]["score"]:
    #                 self[key]["num_contrast_pairs"] += 1
    #                 self[index]["num_contrast_pairs"] += 1
    #                 if not no_info:
    #                     printc('{} contrast pair of {}'.format(index, key), 'magenta')
    #                 self.contrast_pair_num += 1
    #             else:
    #                 if not no_info:
    #                     printc('{} pair of {}'.format(index, key), 'cyan')

    def update_explore(self, index):
        assert index in self.values
        self[index]["num_explore"] += 1

    def max_MC_score_item(self):
        # mean_score = np.mean([self[key]['score'] for key in self.values.keys()])
        # printc('Average Score: {}'.format(str(mean_score)), 'cyan')
        all_scores = {}
        all_score_show = {}
        for index in self.values:
            # score_performance = np.abs(self[index]["score"] - 1 / (1 + np.exp(-64 * (mean_score - 0.5))))
            score_contrast_ratio = np.power(np.max(self[index]['contrast_with_pairs']),
                                            1 / 4)
            score_depth = ((self[index]['score_depth_avg']) / self.depth)
            score_explore = 2 * np.sqrt(1 + self[index]["num_explore"]) / (2 + self[index]["num_explore"])
            # score_explore = np.power(0.8, self[index]["num_explore"])

            # all_scores[index] = 0.3 * score_performance + 0.3 * score_contrast_ratio + 0.4 * score_depth + score_explore
            all_scores[index] = 0.8 * score_contrast_ratio + 0.2 * score_depth + score_explore
            all_score_show[index] = [round(score_contrast_ratio, 2), round(score_depth, 2),
                                     round(score_explore, 2), round(all_scores[index], 2)]
            # 0.125^(1-abs(score-mean)) is 0.125 for 0, about 0.35 for 0.5, and 1 for 1; difference between score 0 and 1 is nearly abs(mean_score-0.5); with 0.5 coefficient
            # np.abs(self[index]["score"] - 1 / (1 + np.exp(-64 * (mean_score - 0.5)))) is nearly 1 for fewer score
            # (num_contrast_pairs/num_pairs)^(1/4) is about 0.55 for 0.1, and 0.75 for 0.3; with 0.5 coefficient
            # 0.8^num_explore change from 1 to 0.1 when num_explore change from 0 to 10: explore one node for 10 times
            # 2 * np.sqrt(1 + self[index]["num_explore"]) / (2 + self[index]["num_explore"]) is about 0.95 for 1, 0.7 for 5, 0.5 for 13 and 0.1 for 396

        printc(str(all_score_show), 'cyan')

        max_value = max(all_scores.values())
        max_index = [key for key, value in all_scores.items() if value == max_value]
        chosen = random.choice(max_index)
        self.update_explore(chosen)
        printc('Chosen {}'.format(chosen), 'background_cyan')
        return chosen

    @staticmethod
    def remove_run_subsequence(index_list_to_detect, explored_result):
        res = []

        def check_success_subsequence(run_index, proposal):
            # run_index succeed. neither longer nor shorter should be run again
            if len(proposal) >= len(run_index):
                return proposal[0:len(run_index)] == run_index
            else:
                return run_index[0:len(proposal)] == proposal

        def check_failed_subsequence(run_index, proposal):
            # run_index failed. shorter should not be run again. longer should be run
            if len(proposal) > len(run_index):
                return False
            else:
                return run_index[0:len(proposal)] == proposal

        def check_subsequence(row, proposal):
            if row['res_cur']:
                return check_success_subsequence(row['run_index'], proposal)
            else:
                return check_failed_subsequence(row['run_index'], proposal)
        for index_new in index_list_to_detect:
            run_detect_res = explored_result.apply(check_subsequence, proposal=index_new, axis=1)
            if run_detect_res.empty:
                res.append(index_new)
            else:
                subsequence_df = explored_result.loc[run_detect_res]
                # Print the 'run_index' values that have the subsequence
                printc('The following run_index is contained in the new sequence:', 'red')
                for index, row in subsequence_df.iterrows():
                    printc(str(row['run_index']), 'red')
        return res

    def propose_explore(self, sequence, max_steps, window_size, explored_result):
        res = []

        def generate_unique_random_lists(N, L, K):
            unique_lists = set()
            for i in range(3 * N):
                # Generate a random list of length L with elements in [0, K]
                random_list = tuple(random.randint(0, K - 1) for _ in range(L))
                unique_lists.add(random_list)
                if len(unique_lists) >= N:
                    break
            # Convert the set of tuples back to a list of lists
            return [list(random_list) for random_list in unique_lists]

        append_index_list = generate_unique_random_lists(2 * window_size, max_steps - len(sequence), self.base)

        # check for index sequences that have been run
        for append_index in append_index_list:
            sequence_new = list(copy.deepcopy(sequence)) + append_index
            assert len(sequence_new) == max_steps
            index_new = tuple(sequence_new)
            res.append(index_new)

        res_new = self.remove_run_subsequence(res, explored_result)

        return res_new

    def next_policy_index_generator(self, explored_result, update_last_k, max_steps, window_size, new_start=False):
        if len(explored_result) == 0:
            res = [tuple([i] + [0] * (max_steps - 1)) for i in range(self.base)]
            res_new = self.remove_run_subsequence(res, explored_result)
            return res_new
        else:
            printc("Current Average Performance: {}".format(explored_result['res_cur'].mean()), 'magenta')

            for _, line in explored_result.tail(update_last_k).iterrows():
                self.update_node_score(line['run_index'], line['res_cur'])

            if new_start:
                res = [tuple([i] + [0] * (max_steps - 1)) for i in range(self.base)]
                res_new = self.remove_run_subsequence(res, explored_result)
                return res_new
            
            else:
                sequence = self.max_MC_score_item()

                proposed = self.propose_explore(sequence, max_steps, window_size, explored_result)

                if window_size > 0:
                    chosen = random.sample(proposed, min(len(proposed), window_size))
                else:
                    chosen = proposed
                printc('New Iteration Over {}'.format(chosen), 'background_magenta')
                return chosen


class codeact_agent_tree(codeact_agent, NBase_Tree):
    def __init__(
            self,
            backbone_func,
            model_name,
            max_steps,
            max_turn,
            memory_size,
            init_prompt_dict,
            in_context_number=0,
            in_context_example=[],
            max_context_length=8192,
            allow_history_cut=False,
            stop=None,
            chat_style=True,
            base=3,
            contrastive_method='temperature',
            max_sample_number=10,
            window_size=3,
    ):
        super().__init__(backbone_func=backbone_func,
                         model_name=model_name,
                         max_steps=max_steps,
                         max_turn=max_turn,
                         memory_size=memory_size,
                         init_prompt_dict=init_prompt_dict,
                         in_context_number=in_context_number,
                         in_context_example=in_context_example,
                         max_context_length=max_context_length,
                         allow_history_cut=allow_history_cut,
                         stop=stop,
                         chat_style=chat_style)

        super(codeact_agent, self).__init__(base, {})

        self.base = base
        self.contrastive_method = contrastive_method
        self.max_sample_number = max_sample_number
        self.window_size = window_size

    def run_multi_times(self, memory, temperature=0.0):
        res = []
        for i in range(5):
            res.append(self.run(memory, temperature=temperature))
        return res

    def get_run_index(self, index):
        assert len(self[index]) > 0
        if self.is_root(index):
            return index
        else:
            index_tem = index
            while len(index_tem) >= 2 and self[index_tem[0:-1]]['done']:
                index_tem = index_tem[0:-1]
            return index_tem

    def run_on_leaf(self, problem_input, env, index):
        if self.is_root(index):
            goal = problem_input

            # if self.chat_style:
            memory = self.make_instruction_history(goal, env)
            steps = 0
            self.print_history(memory)
            # else:
            #     init_obs = env._get_obs()
            #     memory = [("Observation", init_obs)]
            #     steps = 0
            #     instruction = self.make_instruction(goal, env)
            #     printc("Instruction: {}".format(instruction), 'yellow')
            #     printc("Init obs: {}".format(init_obs), 'blue')

            done = False
            res = 0
            env.reset()

        else:
            env_reuse = False
            if len(self.parent_node_value(index)) == 0:
                _, env = self.run_on_leaf(problem_input, env, index[0:-1])
                env_reuse = True

            states = self.parent_node_value(index)
            memory = copy.deepcopy(states['memory'])
            done = states['done']
            steps = states['steps']
            res = states['res']

            if not env_reuse:
                env = restore_task(env, memory)

        if not done:
            # memory, done, res, steps will be updated if not exceed context length limitation; else done and res will be set as True and False
            printc("Running leaf {}".format(index), 'magenta')

            # get leaf policy, memory_new, and temperature
            if self.contrastive_method in ['meta_instruct', 'fixed_instruct', 'contrast_instruct']:
                if self.contrastive_method == 'meta_instruct':
                    policy = generate_step_level_policy(self.backbone_func, index[-1], memory, step_level=True)
                elif self.contrastive_method == 'fixed_instruct':
                    policy = generate_step_level_policy(self.backbone_func, index[-1], memory)
                elif self.contrastive_method == 'contrast_instruct':
                    if index[-1] != 0:
                        if len(self[index[0:-1] + (0,)]) == 0:
                            self.run_on_leaf(problem_input, env, index[0:-1] + (0,))
                        states_normal = self[index[0:-1] + (0,)]
                        memory_normal = copy.deepcopy(states_normal['memory'])
                        last_normal_generation = memory_normal[-2]['content']
                    else:
                        last_normal_generation = ''
                    policy = generate_step_level_policy(self.backbone_func, index[-1], memory,
                                                        last_normal_generation, access_to_normal_generation=True)
                else:
                    raise ValueError('Not supported contrastive_method {}'.format(self.contrastive_method))
                memory_new = add_policy(memory, policy, 'end')
                temperature = get_policy_temperature(index[-1])
            elif self.contrastive_method == 'temperature':
                memory_new = memory
                temperature = get_policy_temperature(index[-1])
                policy = str(temperature)
            elif self.contrastive_method == 'sampling':
                memory_new = memory
                temperature = 1.0  # todo: make sure the generation is different by self.run_multi_times and select
                policy = str(temperature)
            else:
                raise ValueError('Not supported contrastive_method {}'.format(self.contrastive_method))

            # Run on the leaf
            if (not self.allow_history_cut) and self.history_exceeded(memory_new):
                printc("Stopped due to context length limitation", 'red')
                log_info = "context length limitation {} exceeded".format(self.max_context_length)
                done = True
                res = 0

            else:
                generation, action_type, action, extra_info = self.run(
                    memory_new, temperature)

                printc("{} {}:{}".format(action_type, steps, generation), 'green')

                env, state, done = self.execute_action(
                    env, action_type, action, extra_info
                )
                printc("State {}: {}".format(steps, state), 'blue')
                print(
                    "Step {}: Is done: {}".format(
                        steps, done
                    )
                )

                memory = self.update(memory, generation, action_type, action, state)
                steps += 1
                res = done
                log_info = ""


        else:
            # inherit memory, done (= True), res, and steps from parent
            policy = ''
            log_info = "parent node has done the task"

        self[index] = {
            "memory": memory,
            "done": done,
            "res": res,
            "steps": steps,
            "policy": policy,
            "log_info": log_info,
        }

        return {
            "res": res,
            "gen": {
                "run_index": self.get_run_index(index),
                "steps": steps,
                "history": memory,
                "policies": self.retrieve_path(index, 'policy'),
                "log_infos": self.retrieve_path(index, 'log_info')
            },
        }, env

    def __call__(self, problem_input, env, previous_result=None, **kwargs):
        self.reset_values({})

        # index_list = [(1, 1, 1, 1, 0, 0, 0, 0, 0, 0)]
        # index_list = [(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)]
        # def generate_random_list(N, P):
        #     return random.choices([0, 1], weights=[1 - P, P], k=N)
        #
        # index_list = [tuple([0] * self.max_steps), (0, 1, 0, 1, 0, 0, 1, 0, 1, 0)]
        # index_list = []
        # for i in range(self.max_steps + 1):
        #     index_list.append(tuple([1] * i + [0] * (self.max_steps - i)))
        columns = ["full_index", "run_index", "res_cur", "steps", "history", "policies", "log_infos"]
        results_df = pd.DataFrame(
            columns=columns)
        mctree = MCValue_MCTree(self.base, self.max_steps)
        if previous_result is None:
            index_queue = mctree.next_policy_index_generator(results_df, 0, self.max_steps, self.window_size)
        else:
            results_df = previous_result[results_df.columns]
            index_queue = mctree.next_policy_index_generator(results_df, len(results_df), self.max_steps,
                                                             self.window_size, new_start=True)

        run_num = 0
        iteration_num = 0
        empty_num = 0

        while mctree.get_contrast_pair_num() < self.max_sample_number and run_num < 1000 and empty_num < 1000:
            for index in index_queue:
                printc('Iteration {} Run number {} Empty number {} Current Sample Number: {}'.format(iteration_num,
                                                                                                     run_num,
                                                                                                     empty_num,
                                                                                                     mctree.get_contrast_pair_num()),
                       'bright_magenta')
                printc('Testing on Index {}'.format(index), 'magenta')
                env.reset()
                cur_output, env = self.run_on_leaf(problem_input, env, index)
                env.free_resource()

                cur_output['gen']['res_cur'] = float(cur_output['res'])
                cur_output['gen']['full_index'] = index
                # double check that current result has not been run
                if results_df['run_index'].isin([cur_output['gen']['run_index']]).any():
                    print(results_df['run_index'].tolist())
                    print(cur_output['gen']['res_cur'])
                    with open('run_index_repeat_case.txt', 'a') as f:
                        f.write(
                            '{}\n\n{}\n\n{}\n\n{}\n\n\n'.format(index, cur_output['gen']['run_index'], index_queue,
                                                                results_df['run_index'].tolist()))
                    continue
                assert not results_df['run_index'].isin([cur_output['gen']['run_index']]).any()
                results_df = pd.concat([results_df, pd.DataFrame([cur_output['gen']])], ignore_index=True)

                run_num += 1

            iteration_num += 1

            index_queue = mctree.next_policy_index_generator(results_df, len(index_queue),
                                                             self.max_steps, self.window_size)
            if len(index_queue) == 0:
                empty_num += 1
            # else:
            #     empty_num = 0

        max_res = results_df['res_cur'].max()
        results_df = results_df.set_index('run_index')  # the index of result is run_index
        index_dict = results_df.to_dict(orient='index')  # the index here is choice of to_dict method

        results_dict = {'res_cur': {}, 'full_index': {}, 'steps': {}, 'history': {}, 'policies': {}, 'log_infos': {}}
        for index, data in index_dict.items():
            for key in results_dict.keys():
                results_dict[key][index] = data[key]

        result_collection = {'res': max_res, 'gen': results_dict}

        return result_collection

import os
import pdb
import re

import math
import numpy as np
import pandas as pd
import torch
# from agentenv.envs import AcademiaEnvClient, AlfWorldEnvClient, BabyAIEnvClient, MazeEnvClient, WordleEnvClient, \
#     MovieEnvClient, \
#     SciworldEnvClient, SheetEnvClient, SqlGymEnvClient, TextCraftEnvClient, TodoEnvClient, WeatherEnvClient, \
#     WebarenaEnvClient, WebshopEnvClient
from datasets import load_from_disk
from torch.utils.data import Dataset, Subset
from torch.utils.data import IterableDataset

from dataset.codeact_env.codeact_env import CodeActEnv
from dataset.codeact_env.env_tools import execute_with_custom_env, tools_dict, tools_instruction_dict
from dataset.convert_agentgym_to_codeact import convert_agentgym_goal_and_initobs, convert_agentgym_state
from dataset.eurus_eval.utils.grader import math_equal

try:
    import sys

    sys.path.append("dataset/agent_environments")
except:
    pass
# from dataset.agent_environments.environment.pddl_env.pddl_env import PDDL
try:
    from dataset.agent_environments.environment.alfworld.alfworld_env_mine import (
        AlfWorld,
        AlfWorld_Reverse_PREFIXES, AlfWorld_PREFIXES
    )
    IMPORT_ALFWORLD = True
except:
    IMPORT_ALFWORLD = False

# from scripts.eval.m3tooleval.tasks import get_task_iterator

dataset_file_map = {
    "gsm": "gsm",
    "penguin": "penguins_in_a_table",
    "colored_object": "reasoning_about_colored_objects",
    "date_understanding": "date_understanding",
    "penguin_h": "penguins_in_a_table",
    "colored_object_h": "reasoning_about_colored_objects",
    "date_understanding_h": "date_understanding",
}


# class PDDLDataset(Dataset):
#     def __init__(self, dataset_name):
#         self.dataset_name = dataset_name
#         directory_path = (
#             "dataset/agent_environments/environment/pddl_env/pddlgym/pddl/{}".format(
#                 dataset_name
#             )
#         )
#         file_count = len(
#             [
#                 name
#                 for name in os.listdir(directory_path)
#                 if os.path.isfile(os.path.join(directory_path, name))
#             ]
#         )
#         self.data = list(range(file_count))
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, index):
#         env = PDDL(problem_index=index, game_name=self.dataset_name)
#         item = {"input": env, "target": None, "index": index}
#         return item

if IMPORT_ALFWORLD:
    class AlfWorldDataset(Dataset):
        def __init__(self, dataset_name):
            self.dataset_name = dataset_name
            directory_path = "dataset/alfworld/json_2.1.1/valid_unseen"
            env_full = AlfWorld(
                "eval_out_of_distribution",
                base_config="dataset/agent_environments/environment/alfworld/base_config.yaml",
                batch_size=1,
                seed=1,
            )
            type_files = []
            for file in env_full.env.gamefiles:
                # the file form is xxx/pick_cool_then_place_in_recepxxx/trial_xxx/game.tw-pddl
                if file.split("/")[-3].startswith(dataset_name):
                    type_files.append(file)
            if len(type_files) == 0:
                raise ValueError("task type {} not found in file".format(dataset_name))

            self.data = type_files

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            # transform the env type from original alfworld to our implementation
            alf_env = AlfWorld(
                "eval_out_of_distribution",
                base_config="dataset/agent_environments/environment/alfworld/base_config.yaml",
                batch_size=1,
                seed=1,
                task_type=self.dataset_name,
                id=index,
            )
            task_goal = alf_env.goal.replace('Your task is to: ', '')

            def custom_env_reset_func(custom_env):
                custom_env.reset()
                return custom_env

            def custom_env_execution_func(custom_env, action):
                res_global_vars = execute_with_custom_env(custom_env, tools_dict['alfworld'], action)
                custom_env = res_global_vars["env"]
                state, reward, done, infos = custom_env._get_obs(), custom_env.reward, custom_env.done, custom_env.infos
                env_info = {'done': done}
                return custom_env, state, env_info

            env = CodeActEnv(
                name='AlfWorldEnv',
                tools={},
                tools_instruction=tools_instruction_dict['alfworld'],
                goal=task_goal,
                action_type='python',
                require_answer=False,
                custom_env=alf_env,
                custom_env_reset_func=custom_env_reset_func,
                custom_env_execution_func=custom_env_execution_func,
                init_obs=alf_env.init_obs
            )
            # test env
            # execution_result=[]
            # execution_result.append(env.execute_action("goto('drawer 1')"))
            # execution_result.append(env.execute_action("open('drawer 1')"))
            # execution_result.append(env.execute_action("toggle('drawer 1')"))
            # print(execution_result)
            task_file = alf_env.file_path
            task_type = [AlfWorld_PREFIXES[key] for key in AlfWorld_PREFIXES if key in task_file]
            if len(task_type) > 0:
                task_type = task_type[0]
            else:
                task_file = 'Unknown'
            item = {"input": task_goal, 'env': env, "target": None, "index": index, "task_file": task_file,
                    "task_type": task_type}
            return item


# client_map = {'academia': AcademiaEnvClient, 'alfworld': AlfWorldEnvClient, 'babyai': BabyAIEnvClient,
#               'maze': MazeEnvClient, 'wordle': WordleEnvClient, 'movie': MovieEnvClient,
#               'sciworld': SciworldEnvClient, 'sheet': SheetEnvClient, 'sqlgym': SqlGymEnvClient,
#               'textcraft': TextCraftEnvClient, 'todo': TodoEnvClient, 'weather': WeatherEnvClient,
#               'webarena': WebarenaEnvClient, 'webshop': WebshopEnvClient}

# dataset_length_map = {'alfworld': AlfWorldEnvClient, 'babyai': 40,
#                       'sciworld': 4639,
#                       'textcraft': TextCraftEnvClient, 'maze': 25, 'wordle': 100,
#                       'academia': AcademiaEnvClient, 'movie': MovieEnvClient, 'sheet': SheetEnvClient,
#                       'todo': TodoEnvClient, 'weather': WeatherEnvClient,
#                       'webarena': WebarenaEnvClient, 'webshop': 6909,
#                       'sqlgym': SqlGymEnvClient}


# class PortDataset(Dataset):
#     def __init__(self, dataset_name, dataset_port_id):
#         self.dataset_name = dataset_name
#         self.dataset_port_id = dataset_port_id
#         self.dataset_length = dataset_length_map[dataset_name]
#
#     def __len__(self):
#         return self.dataset_length
#
#     def __getitem__(self, index):
#         # Run the bash command
#         # cmd = "source activate agentenv-{} && {} --host 0.0.0.0 --port {}".format(self.dataset_name, self.dataset_name,
#         #                                                                          dataset_port_id)
#         # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         # stdout, stderr = process.communicate()
#         #
#         # # Check if there were any errors during the execution of the command
#         # if process.returncode != 0:
#         #     raise RuntimeError(f"Failed to execute command: {cmd}\nError: {stderr.decode()}")
#
#         # Create the BabyAIEnvClient object
#         if index >= self.dataset_length:
#             raise IndexError("index out of range")
#
#         if self.dataset_name in ['maze', 'wordle']:
#             URL = "http://127.0.0.1:{}/{}".format(self.dataset_port_id, self.dataset_name)
#         else:
#             URL = "http://127.0.0.1:{}".format(self.dataset_port_id)
#         client = client_map[self.dataset_name](**{
#             "env_server_base": URL,
#             "data_len": 200,  # not used
#             "timeout": 300,
#         })
#         # Reset the client with the given index
#         client.reset(index)
#         state = client.observe()
#         conversation = list(client.conversation_start)
#
#         # client.step("search [butt lifting light weight women's shorts high waist tummy control black 3x-large price<50.00]")
#         # client.step('search[pillows]')
#         def custom_env_reset_func(custom_env):
#             custom_env.reset(index)
#             custom_env.observe()
#             return custom_env
#
#         task_goal, init_obs, action_type = convert_agentgym_goal_and_initobs(self.dataset_name, state)
#
#         if action_type == 'python':
#             def custom_env_execution_func(custom_env, action):
#                 res_global_vars = execute_with_custom_env(custom_env, tools_dict[self.dataset_name], action)
#                 custom_env = res_global_vars["env"]
#                 state, done = custom_env.info['observation'], custom_env.info['done']
#                 env_info = {'done': done}
#                 return custom_env, state, env_info
#
#         elif action_type == 'text':
#             if self.dataset_name not in ['webshop']:
#                 def custom_env_execution_func(custom_env, action):
#                     custom_env.step(action)
#                     state, done = custom_env.info['observation'], custom_env.info['done']
#                     state = convert_agentgym_state(self.dataset_name, state)
#                     env_info = {'done': done}
#                     return custom_env, state, env_info
#             else:
#                 def custom_env_execution_func(custom_env, action):
#                     stepout = custom_env.step(action)
#                     state, done = stepout.state, stepout.done
#                     state = convert_agentgym_state(self.dataset_name, state)
#                     env_info = {'done': done}
#                     return custom_env, state, env_info
#         else:
#             raise ValueError('action type {} not implied'.format(action_type))
#
#         env = CodeActEnv(
#             name=self.dataset_name,
#             tools={},
#             tools_instruction=tools_instruction_dict[self.dataset_name],
#             goal=task_goal,
#             action_type=action_type,
#             require_answer=False,
#             custom_env=client,
#             custom_env_reset_func=custom_env_reset_func,
#             custom_env_execution_func=custom_env_execution_func,
#             init_obs=init_obs
#         )
#         # env.execute_action('move down')
#         # env.execute_action('move_forward()')
#         item = {"input": task_goal, 'env': env, "target": None, "index": index}
#         return item


# class M3ToolEvalDataset(IterableDataset):
#     def __init__(self):
#         self.task_iterator = get_task_iterator()
#
#     def __iter__(self):
#         # transform the env type from original m3tooleval to our implementation
#         for index, env in enumerate(self.task_iterator):
#             tools_func = {}
#             tools_instruction = {}
#             for name, tool in env.tools.items():
#                 tools_func[name] = tool.function
#                 tools_instruction[name] = {'description': tool.description, 'fn_signature': tool.fn_signature}
#             env_mine = CodeActEnv(
#                 name=env.name,
#                 tools=tools_func,
#                 tools_instruction=tools_instruction,
#                 goal=env.instruction,
#                 action_type='python',
#                 require_answer=True,
#                 default_answer_checker_reference=env.expected_output,
#                 answer_type='number',
#             )
#             item = {"input": env.instruction, "env": env_mine, "target": None, "index": index}
#             yield item


INVALID_ANS = "[invalid]"

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS


def GSM8KEvaluator(model_output, data):
    def extract_answer(completion):
        try:
            last_number = re.findall(r'\d+', completion)[-1]
            return eval(last_number)
        except:
            return INVALID_ANS

    gold = data['target']
    return extract_answer(model_output) == gold


def MATHEvaluator(model_output, data):
    answer = data['target']
    if not isinstance(answer, str):
        answer = str(answer)
    try:
        if "\pi" in model_output or "\pi" in answer:
            equivs = []
            for pi in [math.pi, 3.14]:
                equivs.append(math_equal(model_output, answer, timeout=True, pi=pi))
            equiv = any(equivs)
        else:
            equiv = math_equal(model_output, answer, timeout=True)
    except:
        equiv = False
    return equiv


class GSM8kDataset(Dataset):
    def __init__(self):
        dataset = load_from_disk('dataset/data/eval/gsm8k')
        self.data = dataset["test"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        answer = extract_answer_hf(item['answer'])
        assert answer != INVALID_ANS, "No ground truth answer found in the document."
        Evaluator = lambda x: MATHEvaluator(x, {'target': answer})
        env = CodeActEnv(
            name="PythonInterpreter",
            tools={},
            tools_instruction={},
            goal=item['question'],
            action_type='python',
            require_answer=True,
            default_answer_checker_reference=answer,
            answer_type='number',
            custom_completion_evaluator=Evaluator,
        )

        res = {"input": item['question'], "target": answer, "env": env, "solution": item['answer'], "index": index}
        return res


class MATHDataset(Dataset):
    def __init__(self):
        dataset = pd.read_json(
            os.path.join('dataset', 'eurus_eval', 'Math', 'math', "math_test_cleaned.json")).to_dict(orient="records")
        dataset = dataset
        self.data = dataset[:1400]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        answer = item["expected_answer"]
        Evaluator = lambda x: MATHEvaluator(x, {'target': answer})
        env = CodeActEnv(
            name="PythonInterpreter",
            tools={},
            tools_instruction={},
            goal=item['problem'],
            action_type='python',
            require_answer=True,
            default_answer_checker_reference=answer,
            answer_type='number',
            custom_completion_evaluator=Evaluator,
        )

        res = {"input": item['problem'], "target": answer, "env": env, "solution": answer, "index": index}
        return res


def PDDLEvaluator(model_output, data):
    return float(model_output)


def AlfWorldEvaluator(model_output, data):
    return float(model_output)


def AgentEvaluator(model_output, data):
    return float(model_output)


class Batch_dataset(Dataset):
    def __init__(self, dataset, batch_size=None):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        if self.batch_size is None:
            return 1
        else:
            return math.ceil(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        if index >= self.__len__():
            raise StopIteration
        if self.batch_size is None:
            item_list = [item for item in self.dataset]
        else:
            item_list = [
                item
                for item in self.dataset[
                            (index * self.batch_size): (
                                min(len(self.dataset), (index + 1) * self.batch_size)
                            )
                            ]
            ]
        return item_list


def get_dataset_part(dataset, i, n):
    """
    Split a PyTorch dataset into n parts and get the i-th part.

    Parameters:
    dataset (Dataset): the original dataset.
    i (int): the index of the part to get.
    n (int): the total number of parts.

    Returns:
    Subset: the i-th part of the dataset.
    """
    length = len(dataset)
    indices = list(range(length))
    part_size = math.ceil(length / n)

    if i < n - 1:
        part_indices = indices[i * part_size: (i + 1) * part_size]
    elif i == n - 1:
        # The last part includes all remaining data points.
        part_indices = indices[i * part_size:]
    else:
        part_indices = []

    print('Distributed Testing on Index {}'.format(part_indices))
    return Subset(dataset, part_indices)


def get_dataset(args):
    data_cleaner = None
    test_index_key = "index"

    # if args.dataset_name in [
    #     "baking",
    #     "barman",
    #     "blocks",
    #     "block_medium",
    #     "blockworld",
    #     "doors",
    #     "elevator",
    #     "footwear",
    #     "fridge",
    #     "glibrearrangement",
    #     "gripper",
    #     "hanoi",
    #     "mineraft",
    #     "newspapers",
    #     "spannerlearning",
    #     "strage",
    #     "termes",
    #     "tireworld_test",
    #     "trapnewspapers",
    #     "tyreworld",
    # ]:
    #     dataset = PDDLDataset(args.dataset_name)
    #     evaluator = PDDLEvaluator
    #     test_index_key = "index"


    #
    # elif args.dataset_name in ['m3tooleval']:
    #     dataset = M3ToolEvalDataset()
    #     evaluator = AgentEvaluator
    #     test_index_key = "index"

    if args.dataset_name in ['gsm8k']:
        dataset = GSM8kDataset()
        evaluator = AgentEvaluator
    elif args.dataset_name in ['math']:
        dataset = MATHDataset()
        evaluator = AgentEvaluator

    elif args.dataset_name in ['humaneval']:
        dataset = GSM8kDataset()
        evaluator = AgentEvaluator

    elif args.dataset_name in [
        "alfworld_put",
        "alfworld_clean",
        "alfworld_heat",
        "alfworld_cool",
        "alfworld_examine",
        "alfworld_puttwo",
        "alfworld"
    ]:
        dataset = AlfWorldDataset(
            AlfWorld_Reverse_PREFIXES[args.dataset_name.replace("alfworld_", "")]
        )
        evaluator = AlfWorldEvaluator
        test_index_key = "index"

    # elif args.dataset_name in ['babyai', 'maze', 'wordle', 'sciworld', 'sqlgym', 'textcraft', 'tool', 'webarena',
    #                            'webshop']:
    #     dataset = PortDataset(args.dataset_name, args.dataset_port_id)
    #     evaluator = AgentEvaluator

    else:
        raise ValueError("dataset not implied yet")

    if not isinstance(dataset, dict):  # split is not specified

        if args.split_dataset_num is not None:
            if args.split_dataset_num[0] < 1:
                # if args.split_dataset_num is not None:
                #     print(
                #         'Warning! Both split_dataset_num and split_dataset_ratio specified. Using split_dataset_ratio.')
                args.split_dataset_num = [
                    int(len(dataset) * ratio) for ratio in args.split_dataset_num
                ]
            else:
                args.split_dataset_num = [int(num) for num in args.split_dataset_num]
            # Add the dataset size of remaining data
            if len(args.split_dataset_num) < 3:
                args.split_dataset_num.append(
                    len(dataset) - np.sum(args.split_dataset_num)
                )

            split_file_path = os.path.join(
                "dataset",
                "split",
                "{}_{}_split_{}.csv".format(
                    args.dataset_name, args.split_dataset_num, args.split_file
                ),
            )
            if not os.path.exists(os.path.join("dataset", "split")):
                os.mkdir(os.path.join("dataset", "split"))

            # The case of train-val-test split
            if len(args.split_dataset_num) == 3:

                if os.path.exists(split_file_path):
                    print("Loading existing split file {}".format(split_file_path))
                    split_pd = pd.read_csv(split_file_path, index_col=0)
                    train_ind = split_pd[split_pd["split"] == "train"][
                        "ind"
                    ].values.tolist()
                    val_ind = split_pd[split_pd["split"] == "val"][
                        "ind"
                    ].values.tolist()
                    test_ind = split_pd[split_pd["split"] == "test"][
                        "ind"
                    ].values.tolist()
                else:
                    train_ind, val_ind, test_ind = torch.utils.data.random_split(
                        list(range(len(dataset))), args.split_dataset_num
                    )

                    train_ind = train_ind.indices
                    val_ind = val_ind.indices
                    test_ind = test_ind.indices
                    split_list = []
                    for ind in train_ind:
                        split_list.append([ind, "train"])
                    for ind in val_ind:
                        split_list.append([ind, "val"])
                    for ind in test_ind:
                        split_list.append([ind, "test"])
                    split_pd = pd.DataFrame(split_list, columns=["ind", "split"])
                    split_pd.to_csv(split_file_path, header=True)

                train_set = Subset(dataset, train_ind)
                val_set = Subset(dataset, val_ind)
                test_set = Subset(dataset, test_ind)

                if args.batch_train:
                    train_set = Batch_dataset(train_set)

                if args.test_on_train:
                    dataset = {"train": train_set, "val": val_set, "test": train_set}
                elif args.test_on_all:
                    dataset = {"train": train_set, "val": val_set, "test": dataset}
                else:
                    dataset = {"train": train_set, "val": val_set, "test": test_set}

            else:
                assert len(args.split_dataset_num) == 2

                if os.path.exists(split_file_path):
                    print("Loading existing split file {}".format(split_file_path))
                    split_pd = pd.read_csv(split_file_path, index_col=0)
                    train_ind = split_pd[split_pd["split"] == "train"][
                        "ind"
                    ].values.tolist()
                    test_ind = split_pd[split_pd["split"] == "test"][
                        "ind"
                    ].values.tolist()
                else:
                    train_ind, test_ind = torch.utils.data.random_split(
                        list(range(len(dataset))), args.split_dataset_num
                    )

                    train_ind = train_ind.indices
                    test_ind = test_ind.indices

                    split_list = []
                    for ind in train_ind:
                        split_list.append([ind, "train"])
                    for ind in test_ind:
                        split_list.append([ind, "test"])
                    split_pd = pd.DataFrame(split_list, columns=["ind", "split"])
                    split_pd.to_csv(split_file_path, header=True)

                train_set = Subset(dataset, train_ind)
                test_set = Subset(dataset, test_ind)

                if args.batch_train:
                    train_set = Batch_dataset(train_set)

                if args.test_on_train:
                    dataset = {"train": train_set, "val": None, "test": train_set}
                elif args.test_on_all:
                    dataset = {"train": train_set, "val": None, "test": dataset}
                else:
                    dataset = {"train": train_set, "val": None, "test": test_set}

        else:
            dataset = {"train": None, "val": None, "test": dataset}

    return {
        "dataset": dataset,
        "evaluator": evaluator,
        "data_cleaner": data_cleaner,
        "test_index_key": test_index_key,
    }

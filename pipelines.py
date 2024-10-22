import copy
import json
import os.path
from threading import Thread, BoundedSemaphore, Lock

import pandas as pd
from torch.utils.data import Subset
from tqdm import tqdm

from dataset.get_dataset import get_dataset_part
from sup_func.sup_func import pd_concat_ignore2, eval_pd_dataset, flat_row


def answer_eval(response, dataset, data, evaluator=None, model_result_recoder=None):
    result = copy.copy(data)

    if response["res"] is None:
        score = 0
    else:

        if dataset in [
            "baking",
            "barman",
            "blocks",
            "block_medium",
            "blockworld",
            "doors",
            "elevator",
            "footwear",
            "fridge",
            "glibrearrangement",
            "gripper",
            "hanoi",
            "mineraft",
            "newspapers",
            "spannerlearning",
            "strage",
            "termes",
            "tireworld_test",
            "trapnewspapers",
            "tyreworld",
        ]:
            score = evaluator(response["res"], data)
        elif dataset in [
            "alfworld_put",
            "alfworld_clean",
            "alfworld_heat",
            "alfworld_cool",
            "alfworld_examine",
            "alfworld_puttwo",
            "alfworld"
        ]:
            score = evaluator(response["res"], data)
        elif dataset in ['m3tooleval']:
            score = evaluator(response["res"], data)
        elif dataset in ['gsm8k', 'math']:
            score = evaluator(response["res"], data)
        elif dataset in ['babyai', 'maze', 'wordle', 'sciworld', 'sqlgym', 'textcraft', 'tool', 'webarena', 'webshop']:
            score = evaluator(response["res"], data)
        else:
            print("evaluation of dataset {} not implied yet".format(dataset))
            raise ValueError

    result["score"] = score

    if model_result_recoder is None:
        # default response record items
        result["res"] = response["res"]
        if isinstance(response["gen"], dict):
            for key in response["gen"].keys():
                result[key] = response["gen"][key]
        else:
            result["generation"] = response["gen"]
    else:
        # model specific response record
        result = model_result_recoder(response, result)

    return score, result


def save_result_jsonl(file_name, result):
    with open(file_name, "a") as f:
        f.write(json.dumps(result) + "\n")
        f.flush()


def save_result_pd(file_name, result, sort_columns=False):
    # File index is not used
    df = pd.DataFrame([[result[key]
                        for key in result.keys()]], columns=result.keys())
    if os.path.exists(file_name):
        result_per_dataset_table_permutated_all = pd_concat_ignore2(
            pd.read_csv(file_name, index_col=0), df
        )
    else:
        result_per_dataset_table_permutated_all = df
    if sort_columns:
        result_per_dataset_table_permutated_all = (
            result_per_dataset_table_permutated_all.sort_index(axis=1)
        )
    if os.path.dirname(file_name) != "" and not os.path.exists(
            os.path.dirname(file_name)
    ):
        os.makedirs(os.path.dirname(file_name))
    result_per_dataset_table_permutated_all.to_csv(file_name, header=True)


def update_result_pd(file_name, result, replace_id):
    df = pd.DataFrame([[result[key]
                        for key in result.keys()]], columns=result.keys())
    if os.path.exists(file_name):
        result_df = pd.read_csv(file_name, index_col=0)
        result_df.at[replace_id, "usable"] = False  # Mark for filter
        result_per_dataset_table_permutated_all = pd_concat_ignore2(
            result_df, df)
    else:
        result_per_dataset_table_permutated_all = df
    if os.path.dirname(file_name) != "" and not os.path.exists(
            os.path.dirname(file_name)
    ):
        os.makedirs(os.path.dirname(file_name))
    result_per_dataset_table_permutated_all.to_csv(file_name, header=True)


def resume_result_jsonl(file_name):
    lines = open(file_name).readlines()
    num_skip_exps = len(lines)
    for id, data in enumerate(map(json.loads, lines)):
        if "score" not in data:
            print(id, data)
    scores = [data["score"] for data in map(json.loads, lines)]
    return scores, num_skip_exps


def resume_result_pd(file_name, executed_column):
    resume = pd.read_csv(file_name, usecols=[executed_column])
    if "score" in resume:
        scores = resume["score"].values.tolist()
    elif "success" in resume:
        scores = (resume["success"] > 0).values.tolist()
    else:
        print("Warnning! No score or success in resumed file. No score resumed")
        scores = []
    if executed_column in resume:
        executed_samples = []
        for sample_list in resume[executed_column].values.tolist():
            if not pd.isna(sample_list):
                try:
                    samples = eval(sample_list)
                    if not isinstance(samples, list):
                        executed_samples.append(str(samples))
                    else:
                        for item in eval(sample_list):
                            executed_samples.append(str(item))
                except:
                    executed_samples.append(str(sample_list))
        executed_samples = set(executed_samples)
        num_skip_exps = len(executed_samples)
    else:
        num_skip_exps = len(resume)
        executed_samples = set()
    return scores, num_skip_exps, executed_samples


def resume_result_data_level(file_name, executed_column, distributed_test, distributed_id, distributed_number):
    # specific function for codeact_agent_tree; chaneg each line into an individual df
    if distributed_test:
        total_rows = 0
        for chunk in pd.read_csv(file_name, chunksize=1):
            total_rows += len(chunk)
        print('total line {}'.format(total_rows))
        to_load_ids = get_dataset_part(range(total_rows), distributed_id, distributed_number)
        resume = pd.read_csv(file_name,skiprows=range(1, to_load_ids[0]), nrows=len(to_load_ids))
    else:
        resume = pd.read_csv(file_name)

    resume = eval_pd_dataset(resume)

    if "score" in resume:
        scores = resume["score"].values.tolist()
    elif "success" in resume:
        scores = (resume["success"] > 0).values.tolist()
    else:
        print("Warnning! No score or success in resumed file. No score resumed")
        scores = []

    if executed_column in resume:
        executed_samples = []
        for sample_list in tqdm(resume[executed_column].values.tolist()):
            if not pd.isna(sample_list):
                try:
                    samples = eval(sample_list)
                    if not isinstance(samples, list):
                        executed_samples.append(str(samples))
                    else:
                        for item in eval(sample_list):
                            executed_samples.append(str(item))
                except:
                    executed_samples.append(str(sample_list))

        executed_samples = set(executed_samples)
        num_skip_exps = len(executed_samples)
    else:
        raise ValueError('executed_column {} not in resume'.format(executed_column))
    flat_resume = pd.concat(resume.apply(flat_row, axis=1).tolist(), ignore_index=True)
    return scores, num_skip_exps, executed_samples, flat_resume


def test_single_sample(
        data, model, args, file_name, evaluator, is_parallel=False, ignore_error=False
):
    global scores, f, pbar

    if is_parallel or ignore_error:  # ignore error to release the process of parallel
        try:
            response = model(data)
        except Exception as e:
            print(e)
            response = {
                "res": None,
                "gen": None,
                "error": "0_test_single_sample_{}".format(e),
            }
    else:
        response = model(data)

    if is_parallel:
        lock.acquire()

    score, result = answer_eval(
        response, args.dataset_name, data, evaluator, model.result_recoder
    )

    scores.append(score)
    pbar.set_description(f"Total Score : {100 * sum(scores) / len(scores)}")

    save_result_pd(file_name, result)

    if is_parallel:
        lock.release()
        pool.release()


def learn_single_sample(
        data, model, args, file_name, evaluator, is_parallel=False, ignore_error=False
):
    global scores, pbar

    if is_parallel or ignore_error:  # ignore error to release the process of parallel
        try:
            response_list = model.learn(data, evaluator)
        except Exception as e:
            print(e)
            response_list = [
                {
                    "tool_cases_list": [data],
                    "error": "0_learn_single_sample_{}".format(e),
                    "success": 0,
                }
            ]
    else:
        response_list = model.learn(data, evaluator)

    if is_parallel:
        lock.acquire()

    assert isinstance(response_list, list)

    score_recorded = False
    for response in response_list:
        if "func_id" in response:
            func_id, response = response["func_id"], response["response"]
            if ("pre_version" not in response) or (
                    func_id is not None and response["pre_version"] is None
            ):
                response["pre_version"] = func_id
        else:
            func_id = None

        if "usable" not in response:
            response["usable"] = response["success"] > 0
        if not score_recorded:
            score = float(response["success"])
            scores.append(float(score > 0))
            score_recorded = True

        pbar.set_description(f"Total Score : {100 * sum(scores) / len(scores)}")
        if func_id is not None:
            update_result_pd(file_name, response, func_id)
        else:
            save_result_pd(file_name, response)

    if is_parallel:
        lock.release()
        pool.release()


pool = BoundedSemaphore(4)
lock = Lock()


def training(dataloader, model, args):
    global scores, pbar

    train_index_key = dataloader["test_index_key"]

    OUTPUT_PATH = (
        args.learn_save_path
        if args.learn_save_path is not None
        else f"learn_results/{args.planning_method}/{args.dataset_name}_{args.split_dataset_num[0]}_split_{args.split_file}.csv"
    )

    print("Saving training to {}".format(OUTPUT_PATH))

    if args.resume and os.path.exists(OUTPUT_PATH):
        print("Resuming training from {}".format(OUTPUT_PATH))
        scores, num_skip_exps, executed_samples = resume_result_pd(
            OUTPUT_PATH, "tool_cases_list"
        )
    else:
        num_skip_exps = 0
        scores = []
        executed_samples = set()
        if os.path.exists(OUTPUT_PATH):
            raise ValueError(
                "Learned file exists. Cannot start a new learning. Please rename the learned file {} first.".format(
                    OUTPUT_PATH
                )
            )

    trial = 0
    threads = []

    pbar = tqdm(dataloader["dataset"]["train"])

    print("executed_samples: {}".format(len(executed_samples)))
    for data in pbar:
        trial += 1
        if not args.parallel_learn:
            if (
                    (
                            train_index_key in data
                            and str(data[train_index_key]) in executed_samples
                    )
                    or (
                    isinstance(data, list)
                    and train_index_key in data[0]
                    and set([str(item[train_index_key]) for item in data]).issubset(
                executed_samples
            )
            )
                    or (str(data) in executed_samples)
            ):
                print("skip")
                continue
            if dataloader["data_cleaner"] is not None and (
                    not dataloader["data_cleaner"](data)
            ):
                print("Dirty Data! Skip")
                continue
            learn_single_sample(
                data,
                model,
                args,
                OUTPUT_PATH,
                dataloader["evaluator"],
                ignore_error=args.ignore_error,
            )
        else:
            if (
                    train_index_key in data
                    and str(data[train_index_key]) in executed_samples
            ) or (str(data) in executed_samples):
                print("skip")
                continue
            if dataloader["data_cleaner"] is not None and (
                    not dataloader["data_cleaner"](data)
            ):
                print("Dirty Data! Skip")
                continue
            pool.acquire()
            thread = Thread(
                target=learn_single_sample,
                args=(
                    data,
                    model,
                    args,
                    OUTPUT_PATH,
                    dataloader["evaluator"],
                    True,
                    args.ignore_error,
                ),
            )
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join(300)  # 5 min for each task
        if thread.is_alive():
            print("A job didn't finish within the time limit")

    print(f"Total Score : {100 * sum(scores) / len(scores)}")
    print("Training finished")


def testing(dataloader, model, args):
    global scores, pbar

    trial = 0
    test_index_key = dataloader["test_index_key"]

    OUTPUT_PATH = (
        args.eval_save_path
        if args.eval_save_path is not None
        else f"eval_results/{args.planning_method}.{args.model_name}/{args.dataset_name}.{args.split_dataset_num}_split_{args.split_file}.{args.exp_id}.{args.distributed_id}.csv"
    )

    print("Saving testing to {}".format(OUTPUT_PATH))

    if args.resume_path is not None:
        resume_path = args.resume_path
    else:
        if args.resume_from_merge:
            parts = OUTPUT_PATH.split('.')
            resume_path = '.'.join(parts[:-2] + ['None'] + [parts[-1]])
        else:
            resume_path = OUTPUT_PATH
    if args.resume and os.path.exists(resume_path):
        print("Resuming testing from {}".format(resume_path))
        if args.resume_data_level:
            assert args.resume_path is not None
            print("Resuming testing data-level")
            scores, num_skip_exps, executed_samples, previous_result_data_level = resume_result_data_level(
                resume_path, test_index_key, args.distributed_test, args.distributed_id, args.distributed_number
            )
        else:
            scores, num_skip_exps, executed_samples = resume_result_pd(
                resume_path, test_index_key
            )
        if args.resume_from_merge or args.resume_path is not None:
            if os.path.exists(OUTPUT_PATH):
                raise ValueError(
                    "Eval result file exists. Cannot start a new testing. Please rename the eval result file {} first.".format(
                        OUTPUT_PATH
                    )
                )
    else:
        scores = []
        num_skip_exps = 0
        executed_samples = set()
        if os.path.exists(OUTPUT_PATH):
            raise ValueError(
                "Eval result file exists. Cannot start a new testing. Please rename the eval result file {} first.".format(
                    OUTPUT_PATH
                )
            )

    print("Executed_samples: {}".format(len(executed_samples)))

    dataset = dataloader["dataset"]["test"]
    assert test_index_key == 'index'
    if not args.resume_data_level:
        left_index = [index for index in range(len(dataset)) if str(index) not in executed_samples]
    else:
        # for data-level resume, only run the data that have been run
        left_index = [index for index in range(len(dataset)) if str(index) in executed_samples]
    dataset = Subset(dataset, left_index)

    if not args.resume_data_level:
        if args.distributed_test:
            dataset = get_dataset_part(dataset, args.distributed_id, args.distributed_number)

    pbar = tqdm(dataset)
    threads = []
    for data in pbar:
        trial += 1
        if not args.parallel_test:
            # if str(data[test_index_key]) in executed_samples or (
            #         str(data[test_index_key]).replace(
            #             "\r\n", "\n") in executed_samples
            # ):
            #     print("skip")
            #     continue
            if dataloader["data_cleaner"] is not None and (
                    not dataloader["data_cleaner"](data)
            ):
                print("Dirty Data! Skip")
                continue

            if args.resume_data_level:
                previous_result = previous_result_data_level[
                    previous_result_data_level[test_index_key] == data[test_index_key]]
                assert len(previous_result) > 0
                data['previous_result'] = previous_result

            test_single_sample(
                data,
                model,
                args,
                OUTPUT_PATH,
                dataloader["evaluator"],
                ignore_error=args.ignore_error,
            )
        else:
            # if (
            #         str(data[test_index_key]) in executed_samples
            #         or (str(data[test_index_key]).replace("\r\n", "\n")) in executed_samples
            # ):
            #     print("skip")
            #     continue
            if dataloader["data_cleaner"] is not None and (
                    not dataloader["data_cleaner"](data)
            ):
                print("Dirty Data! Skip")
                continue
            pool.acquire()
            thread = Thread(
                target=test_single_sample,
                args=(
                    data,
                    model,
                    args,
                    OUTPUT_PATH,
                    dataloader["evaluator"],
                    True,
                    args.ignore_error,
                ),
            )
            threads.append(thread)
            thread.start()

    for thread in threads:
        thread.join(300)  # 5 min for each task
        if thread.is_alive():
            print("A job didn't finish within the time limit")

    print(f"Total Score : {100 * sum(scores) / len(scores)}")
    print("Testing finished")

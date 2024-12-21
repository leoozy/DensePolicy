import pdb

import tiktoken
from transformers import AutoTokenizer
import os
tiktoken_cache_dir = "./cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir


def num_tokens_string(string, model="gpt-3.5-turbo-0613"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        if (
                "gpt-3.5-turbo" in model
                or "gpt-35-turbo" in model
                or "gpt-3.5" in model
                or "gpt-35" in model
        ):
            model = "gpt-3.5-turbo-0613"
        elif "gpt-4" in model:
            print(
                "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
            )
            model = "gpt-4-0613"
        else:
            print("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
        encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(string))


GLOBAL_HF_TOKENIZER = None


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    print(f"model is {model}")
    global GLOBAL_HF_TOKENIZER
    if 'gpt' in model:
        """Return the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_message = 4
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif (
                "gpt-3.5-turbo" in model
                or "gpt-35-turbo" in model
                or "gpt-3.5" in model
                or "gpt-35" in model
        ):
            print(
                "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
            )
            return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
        elif "gpt-4" in model:
            print(
                "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
            )
            return num_tokens_from_messages(messages, model="gpt-4-0613")
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    elif 'Llama' in model:
        if GLOBAL_HF_TOKENIZER is None:
            if '8B' in model:
                GLOBAL_HF_TOKENIZER = AutoTokenizer.from_pretrained(
                    "/storage/home/westlakeLab/zhangjunlei/models/Meta-Llama-3.1-8B-Instruct/llama3.1_8b_intruct/Meta-Llama-3.1-8B-Instruct")
            elif "70B" in model:
                GLOBAL_HF_TOKENIZER = AutoTokenizer.from_pretrained(
                    "/storage/home/westlakeLab/zhangjunlei/models/Meta-Llama-3.1-70B-Instruct")
            else:
                raise NotImplementedError()
        formatted_prompt = GLOBAL_HF_TOKENIZER.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        num_tokens = len(formatted_prompt)
        print(num_tokens)
    elif 'Qwen2.5' in model:
        if GLOBAL_HF_TOKENIZER is None:
            if '7B' in model:
                GLOBAL_HF_TOKENIZER = AutoTokenizer.from_pretrained(
                    "/storage/home/westlakeLab/zhangjunlei/models/Meta-Llama-3.1-8B-Instruct/llama3.1_8b_intruct/Meta-Llama-3.1-8B-Instruct")
            else:
                raise NotImplementedError()
        formatted_prompt = GLOBAL_HF_TOKENIZER.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        num_tokens = len(formatted_prompt)
        print(num_tokens)
    elif 'Mistral' in model:
        if GLOBAL_HF_TOKENIZER is None:
            if "7B" in model:
                GLOBAL_HF_TOKENIZER = AutoTokenizer.from_pretrained(
                    "/storage/home/westlakeLab/zhangjunlei/models/Mistral-7B-Instruct-v0.1/mistralai/Mistral-7B-Instruct-v0.1")
            else:
                raise NotImplementedError()
        formatted_prompt = GLOBAL_HF_TOKENIZER.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        num_tokens = len(formatted_prompt)
        print(num_tokens)
    else:
        raise NotImplementedError(model)

    return num_tokens


def get_max_context_length(model_name):
    if "gpt" in model_name:
        if 'gpt-4' in model_name:
            return 8192
        elif model_name in ["gpt-35", "gpt-3.5", "gpt-35-turbo", "gpt-3.5-turbo"]:
            # set the value according to your api type
            # return 8192
            return 4096
        else:
            raise ValueError(
                "Unimplied max length for model size {}".format(model_name))
    # elif model_name == 'Meta-Llama-3.1-405B-Instruct':
    #     return 4096
    elif "Llama-3" in model_name or "Qwen2.5" in model_name or "Mistral" in model_name:
        return 8192
    # elif "Mistral" in model_name or 'Llama' in model_name or 'deepseek' in model_name:
    #     return 4096
    else:
        raise ValueError(
            "Unimplied max length for base model name {}".format(model_name))

import copy
import os
import time

import numpy as np
import openai

def call_chat_gpt(
        messages,
        hist_messages=None,
        model_name="gpt-3.5-turbo",
        stop=None,
        temperature=0.0,
        top_p=1.0,
        max_tokens=None,
        time_limit_query=200,
        engine_flag=True
):
    if engine_flag:
        openai.api_type = 'azure'
        openai.api_base = 'https://cs-icdevai03-openai-penk-cae.openai.azure.com/'
        openai.api_version = '2023-05-15'
        if model_name == "gpt-3.5":
            model_name = "gpt-35-turbo"
        if model_name == "gpt-35":
            model_name = "gpt-35-turbo"
        elif model_name == "gpt-3.5-turbo":
            model_name = "gpt-35-turbo"

        if os.path.exists("openai_keys_engine.txt"):
            with open("openai_keys_engine.txt") as f:
                gpt_api_key = f.read()
        openai.api_key = gpt_api_key

    else:
        if model_name == "gpt-3.5":
            model_name = "gpt-3.5-turbo"
        if model_name == "gpt-35":
            model_name = "gpt-3.5-turbo"

        if os.path.exists("openai_keys.txt"):
            with open("openai_keys.txt") as f:
                gpt_api_key = f.read()
        openai.api_key = gpt_api_key

    messages = copy.deepcopy(messages)
    hist_messages = copy.deepcopy(hist_messages)
    if hist_messages is not None:
        messages = hist_messages + messages

    wait = 3
    call_time = 0
    save_message = True
    while call_time < 10000:
        try:
            if engine_flag:
                ans = openai.ChatCompletion.create(
                    engine=model_name,
                    max_tokens=max_tokens,
                    stop=stop,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    # n=1
                )
            else:
                ans = openai.ChatCompletion.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    stop=stop,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    # n=1
                )
            content = ans.choices[0].message["content"]
            messages.append(
                {"role": "assistant", "content": content}
            )
            return content, messages

        except Exception as e:
            print(e)
            # time.sleep(min(wait, 50+10*np.random.rand()))
            time.sleep(np.random.rand())
            wait += np.random.rand()
            call_time += 1
    raise RuntimeError("Failed to call gpt")


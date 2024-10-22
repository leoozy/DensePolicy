from model.codeact_agent.in_context_example.gsm8k import gsm8k_codeact_prompt_list, gsm8k_codeact_prompt_list_complete
from model.codeact_agent.in_context_example.math import math_codeact_prompt_list


def get_codeact_in_context_example(task):
    if task in ['gsm8k']:
        return gsm8k_codeact_prompt_list
    elif task in ['math']:
        return math_codeact_prompt_list
    else:
        return []

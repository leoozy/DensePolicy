from typing import Mapping

from sup_func.sup_func import execute


def wrap_tools_with_env(tools_func, custom_env):
    global_vars = {"env": custom_env}
    tools_env = tools_func
    # pass env variable to the tools
    if callable(tools_env):
        tools_env = tools_env(custom_env)
    assert isinstance(tools_env, Mapping)
    for tool_name, tool_function in tools_env.items():
        global_vars[tool_name] = tool_function
    return global_vars


def execute_with_custom_env(custom_env, tools_func, action):
    global_vars = wrap_tools_with_env(tools_func, custom_env)
    res_global_vars = execute(action, global_env=global_vars, time_limit_query=10)
    return res_global_vars

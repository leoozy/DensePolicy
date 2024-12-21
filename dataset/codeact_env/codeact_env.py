import ast
import re
from enum import Enum
from typing import Optional, Mapping, Any, List, Union, Callable

from .repl import PythonREPL
from .repl_mine import PythonExec
from .repl_subprocess import PythonREPL_subprocess


class ActionMode(Enum):
    TEXT_AS_ACTION = "text_as_action"
    JSON_AS_ACTION = "json_as_action"
    CODE_AS_ACTION = "code_as_action"


class CodeActEnv:
    def __init__(
            self,
            name: str,
            tools: Union[Mapping[str, Callable], Callable],
            tools_instruction: Mapping[str, Mapping],
            goal: str,
            action_type: str,
            require_answer: bool,
            answer_type=None,
            default_answer_checker_reference=None,
            custom_completion_evaluator=None,
            custom_env=None,
            custom_env_reset_func=None,
            custom_env_execution_func=None,
            init_obs=None,

    ):
        # name is only used for illustration
        self.name = name

        # tools is only used for default python executor.
        self.tools = tools
        # tools_instruction is used in agent for instruction generation
        self.tools_instruction = tools_instruction

        # goal is currently not used in agent, where the goal is directly passed as input
        self.goal = goal

        # action_type is used in agent (for instruction generation)
        self.action_type = action_type

        # require_answer is used in agent (for instruction generation and completion judgement)
        self.require_answer = require_answer
        # answer_type is used in agent (for instruction generation. It is only required when require_answer is True.)
        self.answer_type = answer_type
        # default_answer_checker_reference is only used when require_answer is True and custom_completion_evaluator is not given. It will be used in the default string matching evaluator
        self.default_answer_checker_reference = default_answer_checker_reference

        # custom method for check answer or task completion.
        self.custom_completion_evaluator = custom_completion_evaluator

        self.custom_env = custom_env
        # if custom_env is given, custom_env_reset_func and custom_env_execution_func must be given.
        self.custom_env_reset_func = custom_env_reset_func
        self.custom_env_execution_func = custom_env_execution_func

        # init_obs is used in agent. If not None, means the task have an initial observation, which need to be given in the first turn.
        self.init_obs = init_obs

        self.print_task()
        self.reset()

    def reset(self) -> None:
        # if tools is a function, call it to get the tools
        if self.custom_env is None:
            if callable(self.tools):
                self.tools = self.tools()
            assert isinstance(self.tools, Mapping)
            self._ns = {tool_name: tool_function for tool_name, tool_function in self.tools.items()}
            self.free_resource()
            self.repl = None
        else:
            self.custom_env = self.custom_env_reset_func(self.custom_env)

    def print_task(self) -> None:
        print('=' * 30)
        print(f"CodeActEnv: {self.name}")
        print('-' * 30)
        print(f"Goal: {self.goal}")
        print('-' * 30)
        print(f"Expected Output: {self.default_answer_checker_reference}")

    def check_answer(self, answer: str, env_info: dict) -> bool:
        # Check if the answer is correct
        if self.custom_completion_evaluator is not None:
            if env_info is not None:
                # The case where the correctness need to be given by env_info
                return bool(self.custom_completion_evaluator(env_info))
            else:
                return bool(self.custom_completion_evaluator(answer))
        else:
            if env_info is not None and 'done' in env_info:
                return env_info['done']
            else:
                try:
                    # Directly compare the answer
                    if answer == self.default_answer_checker_reference or \
                            (self._try_to_convert_to_correct_type(answer)
                             == self._try_to_convert_to_correct_type(self.default_answer_checker_reference)):
                        return True

                    answer = ast.literal_eval(answer)
                    if answer == self.default_answer_checker_reference \
                                (isinstance(self.default_answer_checker_reference, list) and CodeActEnv.compare_list(
                                answer,
                                self.default_answer_checker_reference)) \
                            or (self._try_to_convert_to_correct_type(answer)
                                == self._try_to_convert_to_correct_type(self.default_answer_checker_reference)):
                        return True
                except:
                    pass

                if str(answer) == str(self.default_answer_checker_reference):
                    return True
                return False

    @staticmethod
    def compare_list(a: List[Any], b: List[Any]) -> bool:
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] != b[i] and \
                    (CodeActEnv._try_to_convert_to_correct_type(a[i])
                     != CodeActEnv._try_to_convert_to_correct_type(b[i])):
                return False
        return True

    @staticmethod
    def _try_to_convert_to_correct_type(s: str) -> Any:
        # try int, then float, then str
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s

    def execute_action(self, action: str) -> (Optional[str], dict):
        if self.custom_env is not None:
            try:
                self.custom_env, state, env_info = self.custom_env_execution_func(self.custom_env, action)
                return state, env_info

            except Exception as e:
                state = str(e)
                env_info = {'done': False}
                return state, env_info

        else:
            # execute with python interpreter
            # if action_mode == ActionMode.CODE_AS_ACTION:
            if not self.repl:
                # if self._ns:
                #     self.repl = PythonExec(self._ns)
                # else:
                self.repl = PythonREPL_subprocess()

            # directly execute the code
            obs = self.repl(action)
            # Extract the observation ONLY (remove initial 'Out[0]: ')
            obs = re.sub(r"Out\[\d+\]:", "", obs)

            env_info = None

            try:
                return ast.literal_eval(obs.strip()), env_info
            except Exception as e:
                return obs, env_info

    def free_resource(self) -> None:
        if hasattr(self, 'repl'):
            del self.repl
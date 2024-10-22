import copy
import json
import random
import re
from typing import Optional, Mapping, Any

from backbone.num_tokens import num_tokens_from_messages
from model.codeact_agent.codeact_agent_prompt import get_codeact_instruction
from sup_func.sup_func import printc


def postprocess_fn(generation: str) -> str:
    generation = generation.lstrip()
    # Regex pattern to find "Answer:" or "Action:"
    pattern_answer_action = r"(Answer:|Action:)"
    matches_answer_action = list(re.finditer(pattern_answer_action, generation))

    # If no matches or only one match of Answer/Action, return the original string
    if len(matches_answer_action) <= 1:
        return generation

    # Get the index of the start of the second match of Answer/Action
    second_match_start = matches_answer_action[1].start()

    # Trim the string to end before the second match of Answer/Action
    trimmed_generation = generation[:second_match_start].strip()

    # Find the index of the end of the first "Action:"
    first_action_end_index = trimmed_generation.find("Action:") + len("Action:")
    # Check for the next occurrence of "End Action" after the first "Action:"
    end_action_index = trimmed_generation.find("End Action", first_action_end_index)
    # Determine where to trim the string
    if end_action_index != -1:
        # Trim the string to the determined index
        trimmed_generation = trimmed_generation[:end_action_index + len("End Action")].strip()

    # Check for the next occurrence of "Thought:" after the first "Action:"
    next_thought_index = trimmed_generation.find("Thought:", first_action_end_index)
    if next_thought_index != -1:
        trimmed_generation = trimmed_generation[:next_thought_index].strip()

    return trimmed_generation


def parse_generation(generation: str) -> Optional[Mapping[str, Any]]:
    if "Answer:" in generation and "Action:" in generation:
        return {
            "type": "invalid",
            "content": generation,
            "extra_info": "Invalid generation. Your output should contain either 'Action:' or 'Answer:', but not both."
        }

    if "Answer:" in generation:
        # find the first answer
        if generation.count("Answer:") > 1:
            # get the first answer
            answer = generation.split("Answer:")[1].strip()
            extra_info = "You have output more than one answer. Only the first answer will be used."
        else:
            answer = generation[generation.find("Answer:") + len("Answer:"):].strip()
            extra_info = None
        return {
            "type": "answer",
            "content": answer,
            "extra_info": extra_info
        }
    elif "Action:" in generation:
        if generation.count("Action:") > 1:
            action = generation.split("Action:")[1].lstrip()
            extra_info = "You have output more than one action. Only the first action will be used."
        else:
            action = generation[generation.find("Action:") + len("Action:"):].lstrip()
            extra_info = None

        if "End Action" in action:  # remove the "End Action" part
            action = action[:action.find("End Action")]

        return {
            "type": "action",
            "content": action,
            "extra_info": extra_info
        }
    else:
        return {
            "type": "invalid",
            "content": generation,
            "extra_info": "Invalid generation. Your output should contain either 'Action:' or 'Answer:'"
        }


class codeact_agent:
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
            examples_only=False

    ):
        self.backbone_func = backbone_func
        self.model_name = model_name

        self.max_steps = max_steps
        self.max_turn = max_turn
        assert (self.max_steps is None) != (self.max_turn is None)

        self.init_prompt_dict = init_prompt_dict
        self.in_context_number = in_context_number
        self.in_context_example = in_context_example
        self.split = {
            "example": [""],
            "text": [""],
            "rule": [""],
            "system_msg": [""],
            "instruction": [""],
            "goal": [""],
        }

        if not chat_style and memory_size is None:
            self.memory_size = max_steps + 10
        else:
            self.memory_size = memory_size

        self.max_context_length = max_context_length
        self.allow_history_cut = allow_history_cut
        self.stop = stop
        self.chat_style = chat_style
        self.examples_only = examples_only

    def make_instruction_history(self, goal, env):
        examples_only = False
        instruction = get_codeact_instruction(env.tools_instruction, env.require_answer, env.answer_type,
                                              env.action_type)
        examples = random.sample(self.in_context_example, self.in_context_number)

        query = ""
        if not self.examples_only:
            query += instruction.strip()
        memory = [
            {
                "role": "user",
                "content": query
            },
        ]
        if len(examples) > 0:
            if not self.examples_only:
                memory[-1]['content'] += "\nHere are examples:\n"
            for example in examples:
                assert memory[-1]['role'] == 'user'
                memory[-1]['content'] += example[0].strip()
                for step in example[1:]:
                    assert isinstance(step, list)
                    assert len(step) == 2
                    memory.append({'role': 'assistant', 'content': step[0].strip()})
                    memory.append({'role': 'user', 'content': step[1].strip()})
                if not self.examples_only:
                    memory[-1][
                        'content'] += "\n\n\nNow, let's move on to the next task. The instructions for this task are the same as the previous ones.\n"
                else:
                    memory[-1][
                        'content'] += "\n\n\n"
        assert memory[-1]['role'] == 'user'
        if env.init_obs is not None:
            memory[-1]['content'] += '\n' + 'Here is the environment state: ' + env.init_obs + '\n'
        memory[-1]['content'] += "You should perform actions to accomplish the goal: " + goal.strip()

        return memory

    def get_instruction_with_history(
            self, memory
    ):
        # if self.chat_style:
        # instruction is already memory[0], so not used here
        messages = copy.deepcopy(memory)
        # if self.allow_history_cut:
        #     num_of_tokens = num_tokens_from_messages(messages, self.model_name)
        #     while num_of_tokens > self.max_context_length:
        #         if len(messages) > 3:
        #             printc("Warning! history are reduced due to length limitation", 'red')
        #             messages = [messages[0]] + messages[3:]
        #         else:
        #             raise ValueError("History can not be further reduced")
        #         num_of_tokens = num_tokens_from_messages(messages, self.model_name)
        # else:
        #     # do not use this, as the controlling logic has errors
        #     # memory[0] is only obs_0, and instruction need to be concat at the beginning
        #     system_message = self.init_prompt_dict["system_msg"]
        #
        #     history = memory[-self.memory_size:]
        #     input_prompt = instruction + \
        #                    "\n".join([item[0] + ": " + item[1] for item in history])
        #
        #     messages = [
        #         {"role": "system", "content": system_message},
        #         {"role": "user", "content": input_prompt},
        #     ]
        #     if self.allow_history_cut:
        #         num_of_tokens = num_tokens_from_messages(messages, self.model_name)
        #         while num_of_tokens > self.max_context_length:
        #             print("Warning! history are reduced due to length limitation")
        #             history = history[1:]
        #             input_prompt = instruction + "\n".join(
        #                 [item[0] + ": " + item[1] for item in history]
        #             )
        #
        #             messages = [
        #                 {"role": "system", "content": system_message},
        #                 {"role": "user", "content": input_prompt},
        #             ]
        #             num_of_tokens = num_tokens_from_messages(messages, self.model_name)

        return messages

    def history_exceeded(self, memory):
        # with open('history_exceeded.json', 'w') as f:
        #     json.dump(memory, f, indent=2)
        messages = self.get_instruction_with_history(memory)
        return num_tokens_from_messages(messages, self.model_name) >= self.max_context_length - 120

    def run(self, memory, temperature=0):
        messages = self.get_instruction_with_history(memory)

        human_do = False
        if not human_do:
            generation, _ = self.backbone_func(messages, stop=self.stop, temperature=temperature)

        action_type, action, extra_info = self.action_parser(generation)

        return generation, action_type, action, extra_info

    @staticmethod
    def action_parser(generation):
        # generation = postprocess_fn(generation)
        parsed = parse_generation(generation)
        action_type = parsed["type"]
        action = parsed["content"]
        if "extra_info" in parsed and parsed["extra_info"] is not None:
            extra_info = parsed["extra_info"]
        else:
            extra_info = ""
        return action_type, action, extra_info

    @staticmethod
    def execute_action(env, action_type, action, extra_info, raise_error=False):
        is_done = False

        if action_type == "action":
            execution_result, env_info = env.execute_action(action)

            content = str(execution_result)
            if extra_info != "":
                content += "\n*Extra reminder: " + extra_info

            if not env.require_answer:  # Check completion at each step
                is_done = env.check_answer(execution_result, env_info)
                if is_done:
                    content += " Your finished the task."

        elif action_type == "answer":
            is_done = env.check_answer(action, None)

            print("Expected output:", env.default_answer_checker_reference)
            print("Is correct:", is_done)
            if is_done:
                content = "Your answer is correct."
            else:
                content = "Your answer is incorrect."
                if extra_info != "":
                    content += "\n*Extra reminder: " + extra_info
        else:
            assert action_type == 'invalid'
            content = extra_info

        return env, content, is_done

    def update(self, memory, generation, action_type, action, state):
        if self.chat_style:
            memory.append({
                "role": "assistant",
                "content": generation
            })
            memory.append({
                "role": "user",
                "content": state
            })
        else:
            memory.append((action_type, action))
            memory.append(("Observation", state))
        return memory

    @staticmethod
    def print_history(memory):
        printc("Instruction:\n" + memory[0]['content'], 'yellow')
        if len(memory) >= 1:
            for step in memory[1:]:
                if step['role'] == 'assistant':
                    printc('Action: ' + step['content'], 'green')
                else:
                    assert step['role'] == 'user'
                    printc('State: ' + step['content'], 'blue')

    def __call__(self, problem_input, env, **kwargs):
        env.reset()

        goal = problem_input

        # if self.chat_style:
        memory = self.make_instruction_history(goal, env)
        self.print_history(memory)
        # else:
        #     init_obs = env._get_obs()
        #     memory = [("Observation", init_obs)]
        #     instruction = self.make_instruction(goal, env)
        #     printc("Instruction: {}".format(instruction), 'yellow')
        #     printc("Init obs: {}".format(init_obs), 'blue')

        finished = False

        step_id = 0
        turn_id = 0

        while True:
            # for step_id in range(self.max_steps):
            if (not self.allow_history_cut) and self.history_exceeded(memory):
                printc("Stopped due to context length limitation", 'red')
                finished = True
                res_dict = {
                    "res": False,
                    "gen": {
                        "steps": step_id + 1,
                        "turns": turn_id,
                        "history": memory,
                        "log_infos": "context length limitation {} exceeded".format(self.max_context_length)
                    },
                }
                break

            generation, action_type, action, extra_info = self.run(memory)
            printc("{} {} {}: {}".format(action_type, step_id, turn_id, generation), 'green')

            env, state, done = self.execute_action(
                env, action_type, action, extra_info
            )
            printc("State {}: {}".format(step_id, state), 'blue')
            print(
                "Step {} Turn {}: Is done: {}".format(
                    step_id, turn_id, done
                )
            )

            memory = self.update(memory, generation, action_type, action, state)
            step_id += 1
            if action_type == 'answer':
                turn_id += 1

            if done:
                finished = True
                res_dict = {
                    "res": True,
                    "gen": {
                        "steps": step_id + 1,
                        "turns": turn_id,
                        "history": memory,
                        "log_infos": "agent finished the task"
                    },
                }
                break

            if (self.max_steps is not None and step_id >= self.max_steps) or (
                    self.max_turn is not None and turn_id >= self.max_turn):
                break

        env.free_resource()

        if finished:
            return res_dict
        else:
            return {
                "res": False,
                "gen": {
                    "steps": step_id + 1,
                    "turns": turn_id,
                    "history": memory,
                    "log_infos": "step limitation {} exceeded".format(self.max_steps)
                },
            }

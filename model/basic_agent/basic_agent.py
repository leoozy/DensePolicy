from backbone.num_tokens import num_tokens_from_messages


class basic_agent:
    def __init__(
            self,
            backbone_func,
            model_name,
            max_steps,
            memory_size,
            init_prompt_dict,
            max_context_length=8192,

    ):
        self.backbone_func = backbone_func
        self.model_name = model_name

        self.max_steps = max_steps
        self.init_prompt_dict = init_prompt_dict
        self.split = {
            "example": [""],
            "text": [""],
            "rule": [""],
            "system_msg": [""],
            "instruction": [""],
            "goal": [""],
        }

        if memory_size is None:
            self.memory_size = max_steps + 10
        else:
            self.memory_size = memory_size

        self.max_context_length = max_context_length

    def reset(self, goal, init_obs, init_act=None):
        self.goal = goal
        self.init_obs = init_obs
        self.memory = (
            [("Action", init_act), ("Observation", self.init_obs)]
            if init_act
            else [("Observation", self.init_obs)]
        )
        self.steps = 0

    def update(self, action_type, action, state):
        self.steps += 1
        self.memory.append((action_type, action))
        self.memory.append(("Observation", state))

    def make_prompt(
            self,
    ):
        instruction = self.init_prompt_dict["instruction"]
        examples = self.init_prompt_dict["examples"]
        system_message = self.init_prompt_dict["system_msg"]
        goal = self.goal,

        query = ""
        query += (
                self.split["instruction"][0] + instruction +
                self.split["instruction"][-1]
        )

        if isinstance(examples, str):
            examples = [examples]

        if len(examples) > 0:
            query += "\nHere are examples:\n" + self.split["example"][0]
            for example in examples:
                query += example + "\n"
            query += self.split["example"][-1]

        query += (
                self.split["goal"][0]
                + "You should perform actions to accomplish the goal: "
                + goal
                + "\n"
                + self.split["goal"][-1]
        )

        history = self.memory[-self.memory_size:]
        input_prompt = query + \
                       "\n".join([item[0] + ": " + item[1] for item in history])

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_prompt},
        ]
        num_of_tokens = num_tokens_from_messages(messages, self.model_name)
        while num_of_tokens > self.max_context_length:
            print("Warning! history are reduced due to length limitation")
            history = history[1:]
            input_prompt = query + "\n".join(
                [item[0] + ": " + item[1] for item in history]
            )

            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_prompt},
            ]
            num_of_tokens = num_tokens_from_messages(messages, self.model_name)

        return messages

    def action_parser_for_special_llms(self, action):
        action = action.split("\n")[0]
        action_type = "Action"
        origin_action = action
        if "action" in action.lower():
            action_temp = action.split("\n")
            for act in action_temp:
                if (
                        "next action" in act and ":" in act
                ):  # zzh: in Claude will return "Here is the next action to take:"
                    idx = action_temp.index(act)
                    while idx + 1 < len(action_temp):
                        if action_temp[idx + 1]:
                            action = action_temp[idx + 1]
                            break
                        idx += 1
                # chang: in case parse tool output
                if act.split(":")[0].lower().endswith("with action input"):
                    action = act
                    break
                if "action" in act.lower() and ":" in act:
                    action_temp = ":".join(act.split(":")[1:])
                    if action_temp != "":
                        action = action_temp
                        break
                if "action" in act.lower() and "is to" in act:
                    action_temp = act.split("is to")[1]
                    if action_temp != "":
                        action = action_temp
                        break

        elif action.split(" ")[0] in ["Think:", "think:", "Think", "think"]:
            action_type = "Think"
            action_temp = ":".join(action.split(":")[1:]).strip()
            action = action_temp

        if action.strip() == "":
            # temperary comment this line for codellama
            action = origin_action.split("\n")[0]
        action = action.strip()
        action = action.split("\n")[0]
        if action_type == "Action":
            action = action.strip(".")
        return action_type, action

    def run(self):
        messages = self.make_prompt()

        human_do = False
        if not human_do:
            action, _ = self.backbone_func(messages, stop=["\n"])

        action_type, action = self.action_parser_for_special_llms(action)

        return action_type, action

    def execute_action(self, env, action_type, action, raise_error=False):
        state, done, infos = env.step(action)
        return env, state, done, infos

    def __call__(self, problem_input, **kwargs):
        env = problem_input

        goal = env._get_goal()
        init_obs = env._get_obs()
        memory = [("Observation", init_obs)]
        steps = 0
        print("Goal: {}".format(goal))
        print("Init obs: {}".format(init_obs))

        done = False

        for step_id in range(self.max_steps):

            action_type, action = self.run()
            print("{} {}:{}".format(action_type, step_id, action))

            env, state, done, infos = self.execute_action(
                env, action_type, action
            )

            print("Step {}: State: {}".format(step_id, state))
            print(
                "Step {}: Is done: {}".format(
                    step_id, done
                )
            )
            self.update(action_type, action, state)
            if done:
                return {
                    "res": env.done,
                    "gen": {
                        "steps": step_id + 1,
                        "history": self.memory,
                    },
                }

        return {
            "res": False,
            "gen": {
                "steps": None,
                "history": self.memory,
            },
        }

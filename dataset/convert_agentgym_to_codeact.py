def convert_agentgym_goal_and_initobs(dataset_name, state):
    if dataset_name in ['babyai']:
        task_goal, init_obs, avalible_action = state.split('\n')
        task_goal = task_goal.replace('Your goal: ', '')
        action_type = 'python'
    elif dataset_name in ['maze', 'wordle']:
        goal_and_init_ob = state.split('\n')[-1]
        init_obs = 'Your current position' + goal_and_init_ob.split('Your current position')[1]
        task_goal = goal_and_init_ob.split('Your current position')[0]
        action_type = 'text'
    elif dataset_name in ['webshop']:
        task_goal = state
        init_obs = None
        action_type = 'text'
    else:
        raise ValueError('Dataset {} not implied'.format(dataset_name))
    return task_goal, init_obs, action_type


def convert_agentgym_state(dataset_name, state):
    if dataset_name in ['webshop']:
        for prefix in ['WebShop [SEP] Instruction: [SEP]', 'Instruction: [SEP]']:
            if prefix in state:
                # Find the position of 'WebShop [SEP] Instruction: [SEP]'
                start_pos = state.find(prefix)
                if start_pos == 0:
                    # Add the length of the target string to the start position
                    start_pos += len(prefix)

                    # Find the position of the next [SEP] after the start position
                    next_sep_pos = state.find('[SEP]', start_pos)
                    result = state[next_sep_pos:].strip()
                    return result
        # prefixes do not exist or are not at beginning
        return state
    else:
        return state

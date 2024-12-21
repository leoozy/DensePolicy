def get_codeact_instruction(tool_descriptions, require_answer, answer_type, action_type) -> str:
    res = ''
    if len(tool_descriptions) > 0:
        tool_desc = "You have access to the following tools:\n"
        for i, (tool_name, tool_description) in enumerate(tool_descriptions.items()):
            tool_desc += f"[{i + 1}] {tool_name}: {tool_description['description']}\n"
            if 'fn_signature' in tool_description:
                tool_desc += f"    Signature: {tool_description['fn_signature']}\n"

        res = tool_desc + "\n"

        # Here assumpt the tasks require answer are coding-style task, and the tasks do not require answer are robotic-style task
        # Currently non-python code type action is only considered for task that do not require_answer
        if require_answer:
            res += "You can use the tools by outputing a block of Python code that invoke the tools.\n"
            res += "You may use for-loops, if-statements, and other Python constructs when necessary.\n"
            res += "Be sure to print the final answer at the end of your code.\n"
            res += "You should begin your tool invocation with 'Action:' and end it with 'End Action'.\n"
            res += "Example: 'Action:\ntool_name(argument_1)\nEnd Action'\n"
        else:
            if action_type == 'python':
                res += "You can use the tools by outputing a line of Python code that invoke the tools.\n"
                res += "You should begin your tool invocation with 'Action:' and end it with 'End Action'.\n"
                res += "Example: 'Action:\ntool_name(argument_1)\nEnd Action'\n"
                res += "You can only invoke one tool at a time.\n"
            elif action_type == 'text':
                res += "You can use the tools by outputing the tool name with its arguments (if any).\n"
                res += "You should begin your tool invocation with 'Action:' and end it with 'End Action'.\n"
                res += "Example: 'Action: tool_name argument_1 End Action'\n"
                res += "You can only invoke one tool at a time.\n"
            else:
                raise ValueError('Not supported action type {}'.format(action_type))
    else:
        res += "You can output a block of Python code.\n"
        res += "You may use for-loops, if-statements, and other Python constructs when necessary.\n"
        res += "Be sure to print the final answer at the end of your code.\n"
        res += "You should begin your coding with 'Action:' and end it with 'End Action'.\n"
        res += "I will provide you with the possible actions in the next step. You can use them as a reference."
        res += "Example: 'Action:\nvar1=func_name(argument_1)\nEnd Action'\n"

    res = res + "\nNow, let's get started!\n\n"
    # res = res + f"Instruction: {self.instruction}"
    res = res + "You can optionally express your thoughts using natural language before your action. For example, 'Thought: I want to use tool_name to do something. Action: <your action to call tool_name> End Action'.\n"

    if require_answer:
        res = res + "Note that your output should always contain either 'Action:' or 'Answer:', but not both.\n"
        res = res + "When you are done, output the result using 'Answer: your answer'\n"
        if answer_type in ['judge']:
            res = res + "Please ONLY output the answer (e.g., Yes or No), without any other text.\n"
        elif answer_type in ['choice']:
            res = res + "Please ONLY output the answer (e.g., A, B, C, etc.), without any other text.\n"
        elif answer_type in ['number']:
            res = res + "Please ONLY output the answer (e.g., single number), without any other text.\n"
        else:
            raise ValueError(f"Invalid answer type: {answer_type}")

        res += "Note: This is a multi-turn task. You only need to write a few lines of code in each output and stop your generation. Remember to print the variable for yourself to check.\n"
        res += "I will provide you with the print result of the code in each turn, allowing you to continue the task during your turn.\n"
        res += "When you finish the coding and I provide you with the printed answer, you should return the answer by 'Answer:'."
        res += "Remember that an answer with 'Answer:' can only be output without any 'Action:'.\n"
        res += "If I tell you the answer is incorrect, you should revise your code.\n"
    else:
        res += "Now, I will inform you of the observation in the environment and the goal you need to achieve. You are required to tell me the action you should take at each step.\n"
        res += "You need to output your thoughts for the action (optionally) and the action using the provided tools in your output.\n"
        res += "I will update you on the new observation after your action has been executed in the environment.\n"
        res += "Continue this process for multiple turns until you achieve the goal.\n"

    return res


def get_codeact_agent_instruction_dict():
    return {'system_msg': "You are a master in planning."}

def Get_BABYAI_TOOLS(env):
    # pass environment to the action processes
    def turn_left():
        env.step('turn left')

    def turn_right():
        env.step('turn right')

    def move_forward():
        env.step(f'move forward')

    def go_to(obj):
        env.step(f'go to {obj}')

    def pick_up(obj):
        env.step(f'pickup {obj}')

    def toggle():
        env.step(f'toggle')

    def go_through(door):
        env.step(f'go through {door}')

    def toggle_and_go_through(door):
        env.step(f'toggle and go through {door}')

    return {
        'turn_left': turn_left,
        'turn_right': turn_right,
        'move_forward': move_forward,
        'go_to': go_to,
        'pick_up': pick_up,
        'toggle': toggle,
        'go_through': go_through,
        'toggle_and_go_through': toggle_and_go_through,
    }


BABYAI_TOOLS_Instruction = {'turn_left': {
    'description': "Rotate yourself to the left.",
    'fn_signature': 'turn_left() -> None'},
    'turn_right': {
        'description': "Rotate yourself to the right.",
        'fn_signature': 'turn_right() -> None'},
    'move_forward': {
        'description': 'Move one step forward.',
        'fn_signature': 'move_forward() -> None'},
    'go_to': {
        'description': "Navigate to a specific location or object, e.g., go_to('grey key 1').",
        'fn_signature': 'go_to(obj: str) -> None'},
    'pick_up': {
        'description': "Grasp an object, e.g., pick_up('red ball 1'). Once this action is performed, you will be carrying the object.",
        'fn_signature': 'pick_up(obj: str) -> None'},
    'toggle': {
        'description': "Open a closed or locked door directly in front of you. To perform this action, there must be a closed or locked door in your immediate vicinity. If you want to open a locked door, you need to be carrying a key that is of the same color as the locked door.",
        'fn_signature': 'toggle() -> None'},
    'go_through': {
        'description': "Go through the door. To perform this action, the target door must be open.",
        'fn_signature': 'go_through(obj: str) -> None'},
    'toggle_and_go_through': {
        'description': "Go through the door. If the target door is open, you will directly go through it by performing this action . If the door it is closed or locked, by performing this action you will first open it, then go through it. If the target door is locked, you need to be carrying a key that is of the same color as the locked door.",
        'fn_signature': 'toggle_and_go_through(obj: str) -> None'},
}

# BABYAI_TOOLS_Instruction = '''- turn right
#
# - turn left
#
# - move forward
#
# - go to <obj> <id>
#
# - pick up <obj> <id>
#
# - go through <door> <id>: <door> must be an open door.
#
# - toggle and go through <door> <id>: <door> can be a closed door or a locked door. If you want to open a locked door, you need to carry a key that is of the same color as the locked door.
#
# - toggle: there is a closed or locked door right in front of you and you can toggle it.'''

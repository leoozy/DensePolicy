def Get_ALFWORLD_TOOLS(env):
    # pass environment to the action processes
    def goto(recep):
        env.step(f'go to {recep}')

    def take(obj, recep):
        env.step(f'take {obj} from {recep}')

    def put(obj, recep):
        env.step(f'put {obj} in/on {recep}')

    def open(recep):
        env.step(f'open {recep}')

    def close(recep):
        env.step(f'close {recep}')

    def toggle(obj_or_recep):
        env.step(f'toggle {obj_or_recep}')

    def clean(obj, recep):
        env.step(f'clean {obj} with {recep}')

    def cool(obj, recep):
        env.step(f'cool {obj} with {recep}')

    def heat(obj, recep):
        env.step(f'heat {obj} with {recep}')

    def inventory():
        env.step('inventory')

    def examine(obj_or_recep):
        env.step(f'examine {obj_or_recep}')

    def use(obj):
        env.step(f'use {obj}')

    return {
        'take': take,
        'put': put,
        'open': open,
        'close': close,
        'toggle': toggle,
        'clean': clean,
        'cool': cool,
        'heat': heat,
        'inventory': inventory,
        'examine': examine,
        'goto': goto,
        'use': use,
    }


ALFWORLD_TOOLS_Instruction = {'goto': {'description': "Move to a different receptacle, for example, goto('shelf 1')  After you move to a receptacle, you will see what objects are in that receptacle, unless it is closed.",
                                       'fn_signature': 'goto(recep: str) -> None'},
                              'take': {'description': "Take an object from the current receptacle where you are, for example, take('apple 3','drawer 3') You must be at the receptacle to perform this action, and the object you take must exist in the receptacle. After taking the object, it will be with you.",
                                       'fn_signature': 'take(obj: str, recep: str) -> None'},
                              'put': {'description': 'Place an object into the current receptacle where you are. You must be at the receptacle to perform this action, and the object you put must be with you. After put the object, it will be in the receptacle.',
                                      'fn_signature': 'put(obj: str, recep: str) -> None'},
                              'open': {'description':  'Open the receptacle where you are. You must be at the receptacle to perform this action, and the receptacle you open must initially be closed.',
                                       'fn_signature': 'open(recep: str) -> None'},
                              'close': {'description': 'Close the receptacle where you are. You must be at the receptacle to perform this action, and the receptacle you close must initially be open.',
                                        'fn_signature': 'close(recep: str) -> None'},
                              'clean': {'description': "Clean the object using the receptacle. The receptacle should be a place existing in the environment where you can clean things, such as 'sink 1'. You must be at the receptacle to perform this action, and the object you clean must be with you.",
                                        'fn_signature': 'clean(obj: str, recep: str) -> None'},
                              'cool': {'description': "Cool down the object using the receptacle. The receptacle should be a place existing in the environment where you can cool things down, such as 'fridge 3'. You must be at the receptacle to perform this action, and the object you cool must be with you.",
                                       'fn_signature': 'cool(obj: str, recep: str) -> None'},
                              'heat': {'description': "Heat up the object using the receptacle. The receptacle should be a place existing in the environment where you can heat things up, such as 'oven 2' or 'microwave 1'. You must be at the receptacle to perform this action, and the object you heat must be with you.",
                                       'fn_signature': 'heat(obj: str, recep: str) -> None'},
                              'inventory': {'description': 'Check the current situation. You can always check your status with this action.',
                                            'fn_signature': 'inventory() -> None'},
                              'examine': {'description': 'Inspect the object you have or the receptacle you are at. You must be at the receptacle or have the object to perform this action.',
                                          'fn_signature': 'examine(obj_or_recep: str) -> None'},
                              'use': {'description': 'Utilize the object. ',
                                      'fn_signature': 'use(obj: str) -> None'},
                              'toggle': {
                                  'description': 'Turn on the object you have or the receptacle you are at.',
                                  'fn_signature': 'toggle(obj_or_recep: str) -> None'},
                              }

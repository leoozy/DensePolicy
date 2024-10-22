import argparse

from sup_func.sup_func import parse_left_args


def add_port_dataset_args(parser):
    parser.add_argument("--dataset_port_id", type=int, default=None)
    return parser


def add_dataset_args(args, left):
    parser_new = argparse.ArgumentParser()

    if args.dataset_name in [
        "baking",
        "barman",
        "blocks",
        "block_medium",
        "blockworld",
        "doors",
        "elevator",
        "footwear",
        "fridge",
        "glibrearrangement",
        "gripper",
        "hanoi",
        "mineraft",
        "newspapers",
        "spannerlearning",
        "strage",
        "termes",
        "tireworld_test",
        "trapnewspapers",
        "tyreworld",
    ]:
        pass
    elif args.dataset_name in [
        "alfworld_put",
        "alfworld_clean",
        "alfworld_heat",
        "alfworld_cool",
        "alfworld_examine",
        "alfworld_puttwo",
        "alfworld"
    ]:
        pass
    elif args.dataset_name in ['m3tooleval', 'gsm8k','math']:
        pass
    elif args.dataset_name in ['babyai', 'maze', 'wordle', 'sciworld', 'sqlgym', 'textcraft', 'tool', 'webarena',
                               'webshop']:
        args, left_new = parse_left_args(args, left, add_port_dataset_args)
    else:
        raise ValueError("dataset not implied yet")

    return args

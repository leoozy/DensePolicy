import argparse
import pdb

from sup_func.sup_func import parse_left_args


def add_learnact_agent_args(parser):
    parser.add_argument("--learner_method", type=str, default="single")
    parser.add_argument("--learner_method_no_preuse", action="store_true")
    parser.add_argument("--optimizer_parallel_learn", action="store_true")

    parser.add_argument("--user_method", type=str, default="zero_py")
    parser.add_argument("--user_model_name", type=str, default=None)

    parser.add_argument("--retrieve_model", default=None, type=str)
    parser.add_argument("--retrieve_file_path", default=None, type=str)
    parser.add_argument("--retrieve_top_k", default=2, type=int)
    return parser


def add_api_agent_args(parser):
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--memory_size", type=int)
    parser.add_argument("--check_actions", type=str, default=None)
    return parser


def add_py_agent_args(parser):
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--memory_size", type=int)
    parser.add_argument("--check_actions", type=str, default=None)
    return parser


def add_code_as_policy_args(parser):
    parser.add_argument("--hier", action="store_true")
    return parser


def add_reflexion_agent_args(parser):
    parser.add_argument("--learner_method", type=str, default="single")
    parser.add_argument("--learner_method_no_preuse", action="store_true")
    parser.add_argument("--optimizer_parallel_learn", action="store_true")

    parser.add_argument("--user_method", type=str, default="zero_py")
    parser.add_argument("--user_model_name", type=str, default=None)

    parser.add_argument("--use_plan_number", type=int, default=None)

    parser.add_argument("--retrieve_model", default=None, type=str)
    parser.add_argument("--retrieve_file_path", default=None, type=str)
    parser.add_argument("--retrieve_top_k", default=2, type=int)
    return parser


def add_learnact_user_args(parser):
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--memory_size", type=int)
    parser.add_argument("--use_hand_tool", type=str, default=None)
    parser.add_argument("--check_actions", type=str, default=None)
    parser.add_argument("--react", action="store_true")
    parser.add_argument("--example_order", type=str, default="origin_first")
    parser.add_argument("--note_position", type=str,
                        default="before_example")
    parser.add_argument("--no_tool_selfprompt", action="store_true")
    parser.add_argument("--no_tool_description", action="store_true")
    parser.add_argument("--full_tool_subprocess", action="store_true")
    return parser


def add_learnact_learner_args(parser):
    parser.add_argument("--optimizer_do_learn", action="store_true")
    parser.add_argument("--pass_optimizer", action="store_true")
    parser.add_argument("--tool_in_context_style", type=str, default="vanilla")
    parser.add_argument("--get_tool_version", type=str, default="structured")
    parser.add_argument("--get_tool_incontext_version",
                        type=str, default="toy")
    parser.add_argument("--usage_version", type=str, default="individual")
    parser.add_argument("--tool_improve_target", type=str, default="step")
    parser.add_argument("--tool_improve_version", type=str, default="both")
    parser.add_argument("--tool_improve_in_context_version",
                        type=str, default="toy")
    parser.add_argument("--tool_improve_history", type=str, default="single")
    parser.add_argument("--step_sample_number", type=int, default=1)
    parser.add_argument("--optimize_iteration_number", type=int, default=1)
    parser.add_argument("--score_type", type=str, default="step_correction")
    parser.add_argument("--same_usage", action="store_true")
    return parser


def add_reflexion_learner_args(parser):
    parser.add_argument("--optimizer_do_learn", action="store_true")
    parser.add_argument("--pass_optimizer", action="store_true")
    parser.add_argument("--optimize_iteration_number", type=int, default=1)
    return parser


def add_codeact_agent_args(parser):
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_turn", type=int, default=None)
    parser.add_argument("--allow_history_cut", action='store_true')
    parser.add_argument("--memory_size", type=int)
    parser.add_argument("--in_context_number", type=int, default=0)
    parser.add_argument("--examples_only", action='store_true')
    return parser


def add_codeact_agent_tree_args(parser):
    parser.add_argument("--contrastive_method", type=str)
    parser.add_argument("--max_sample_number", type=int)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--base", type=int)
    return parser


def add_model_args(args, left):
    if args.planning_method in ["api_agent"]:
        args, left_new = parse_left_args(args, left, add_api_agent_args)

    elif args.planning_method in ["py_agent", "react_py_agent"]:
        args, left_new = parse_left_args(args, left, add_py_agent_args)
        if args.check_actions is not None:
            args.check_actions = "check valid actions"

    elif args.planning_method in ["code_as_policy"]:
        args, left_new = parse_left_args(args, left, add_code_as_policy_args)

    elif args.planning_method in ["learnact_agent"]:
        args, left_new = parse_left_args(args, left, add_learnact_agent_args)
        if args.learner_method == "learnact_learner":
            args, left_new = parse_left_args(args, left_new, add_learnact_learner_args)
        else:
            raise ValueError("model not implied yet")

        if args.user_method == "learnact_user":
            args, left_new = parse_left_args(args, left_new, add_learnact_user_args)
            if args.check_actions is not None:
                args.check_actions = "check valid actions"
            if args.user_model_name is None:
                args.user_model_name = args.model_name
        else:
            raise ValueError("model not implied yet")
        assert not (args.optimizer_parallel_learn and args.parallel_learn)

    elif args.planning_method in ["reflexion_agent"]:
        args, left_new = parse_left_args(args, left, add_reflexion_agent_args)
        if args.learner_method == "reflexion":
            args, left_new = parse_left_args(args, left_new, add_reflexion_learner_args)
        else:
            raise ValueError("model not implied yet")

        if args.user_method in ["py_agent", "react_py_agent"]:
            args, left_new = parse_left_args(args, left_new, add_py_agent_args)
            if args.check_actions is not None:
                args.check_actions = "check valid actions"
            if args.user_model_name is None:
                args.user_model_name = args.model_name
        else:
            raise ValueError("model not implied yet")

        assert not (args.optimizer_parallel_learn and args.parallel_learn)

    elif args.planning_method in ["codeact_agent"]:

        args, left_new = parse_left_args(args, left, add_codeact_agent_args)
    elif args.planning_method in ["codeact_agent_tree"]:
        args, left_new = parse_left_args(args, left, add_codeact_agent_args)
        args, left_new = parse_left_args(args, left_new, add_codeact_agent_tree_args)
    else:
        raise ValueError("model not implied yet")

    return args

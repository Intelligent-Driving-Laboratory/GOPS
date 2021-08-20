import copy
import datetime
import json
import os

from modules.utils.utils import change_type


def init_args(env, **args):

    if args['policy_func_type'] == 'RNN' or args['policy_func_type'] == 'CNN':
        args['obsv_dim'] = env.observation_space.shape
    else:
        args['obsv_dim'] = env.observation_space.shape[0]

    if args['action_type'] == 'continu':
        args['action_dim'] = env.action_space.shape[0]
        args['action_high_limit'] = env.action_space.high
        args['action_low_limit'] = env.action_space.low
    else:
        args['action_num'] = env.action_space.n
        args['noise_params']['action_num'] = args['action_num']

    # Create save arguments
    if args['save_folder'] is None:
        dir_path = os.path.dirname(__file__)
        dir_path = os.path.dirname(dir_path)
        dir_path = os.path.dirname(dir_path)
        args['save_folder'] = os.path.join(dir_path+'/results/',
                                           args['algorithm'],
                                           datetime.datetime.now().strftime("%m%d-%H%M%S"))
    os.makedirs(args['save_folder'], exist_ok=True)
    os.makedirs(args['save_folder'] + '/apprfunc', exist_ok=True)
    os.makedirs(args['save_folder'] + '/evaluator', exist_ok=True)

    with open(args['save_folder'] + '/config.json', 'w', encoding='utf-8') as f:
        json.dump(change_type(copy.deepcopy(args)), f, ensure_ascii=False, indent=4)
    return args

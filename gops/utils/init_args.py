#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description:
#  Update: 2021.03.05, Shengbo LI (example, can be deleted)



import copy
import datetime
import json
import os
import torch
import warnings

from gops.utils.utils import change_type


def init_args(env, **args):
    # cuda
    if args['enable_cuda']:
        if torch.cuda.is_available():
            args['use_gpu'] = True
        else:
            warning_msg = 'cuda is not available, use CPU instead'
            warnings.warn(warning_msg)
            args['use_gpu'] = False
    else:
        args['use_gpu'] = False

    # sampler
    if args['trainer'] == 'on_sync_trainer':
        args['batch_size_per_sampler'] = args['sample_batch_size'] // args['num_samplers']
        if args['sample_batch_size'] % args['num_samplers'] != 0:
            args['sample_batch_size'] = args['batch_size_per_sampler']*args['num_samplers']
            error_msg = "sample_batch_size can not be exact divided by the number of samplers!"
            raise ValueError(error_msg)
    else:
        args['batch_size_per_sampler'] = args['sample_batch_size']


        # observation dimension
    if len(env.observation_space.shape) == 1:
        args['obsv_dim'] = env.observation_space.shape[0]
    else:
        args['obsv_dim'] = env.observation_space.shape

    if args['action_type'] == 'continu':  # get the dimension of continuous action or the num of discrete action
        args['action_dim'] = env.action_space.shape[0] if len(env.action_space.shape) == 1 else env.action_space.shape
        args['action_high_limit'] = env.action_space.high
        args['action_low_limit'] = env.action_space.low
    else:
        args['action_num'] = env.action_space.n
        args['noise_params']['action_num'] = args['action_num']

    if hasattr(env, 'constraint_dim'):  # get the dimension of constrain
        args['constraint_dim'] = env.constraint_dim

    if args['value_func_type'] == 'CNN_SHARED':
        if hasattr(args, 'policy_func_type'):
            assert args['value_func_type'] == args[
                'policy_func_type'], 'The function type of both value and policy should be CNN_SHARED'
            assert args['value_conv_type'] == args[
                'policy_conv_type'], 'The conv type of value and policy should be the same'
        args['cnn_shared'] = True
        args['feature_func_name'] = 'Feature'
        args['feature_func_type'] = 'CNN_SHARED'
        args['conv_type'] = args['value_conv_type']
    else:
        args['cnn_shared'] = False

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

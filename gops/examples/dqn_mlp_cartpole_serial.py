#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao
#  Description: gym environment, discrete action, cart pole, dqn
#  Update Date: 2021-01-03, Yuxuan JIANG & Guojian ZHAN : implement DQN


import argparse
import copy
import datetime
import json
import os

import numpy as np

from modules.create_pkg.create_alg import create_alg
from modules.create_pkg.create_buffer import create_buffer
from modules.create_pkg.create_env import create_env
from modules.create_pkg.create_evaluator import create_evaluator
from modules.create_pkg.create_sampler import create_sampler
from modules.create_pkg.create_trainer import create_trainer
from modules.utils.utils import change_type


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    # Key Parameters for users
    parser.add_argument('--env_id', type=str, default='gym_cartpole', help='')
    parser.add_argument('--apprfunc', type=str, default='MLP', help='')
    parser.add_argument('--algorithm', type=str, default='DQN', help='')
    parser.add_argument('--trainer', type=str, default='serial_trainer', help='')

    # 1. Parameters for environment
    parser.add_argument('--obsv_dim', type=int, default=None, help='')
    parser.add_argument('--action_dim', type=int, default=1, help='')
    parser.add_argument('--action_num', type=int, default=None, help='Num of discrete actions (similar to gym.spaces.Discrete)')
    parser.add_argument('--action_type', type=str, default='disc', choices=['disc', 'conti'], help='')
    parser.add_argument('--is_render', type=bool, default=False)

    # 2. Parameters for approximate function
    parser.add_argument('--value_func_name', type=str, default='q_value', help='')
    parser.add_argument('--value_func_type', type=str, default=parser.parse_args().apprfunc, help='')
    parser.add_argument('--value_hidden_sizes', type=list, default=[256, 256])
    parser.add_argument('--value_hidden_activation', type=str, default='relu', help='')
    parser.add_argument('--value_output_activation', type=str, default='linear', help='')

    # 3. Parameters for algorithm
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.01, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='')
    parser.add_argument('--distribution_type', type=str, default='ValueDirac')

    # 4. Parameters for trainer
    # Parameters for sampler
    parser.add_argument('--sample_batch_size', type=int, default=256, help='')
    parser.add_argument('--sampler_name', type=str, default='mc_sampler')
    parser.add_argument('--noise_params', type=dict, default=None, help='')
    parser.add_argument('--reward_scale', type=float, default=0.1, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--sample_sync_interval', type=int, default=1, help='')
    # Parameters for buffer
    parser.add_argument('--buffer_name', type=str, default='replay_buffer')
    parser.add_argument('--buffer_warm_size', type=int, default=1000)
    parser.add_argument('--buffer_max_size', type=int, default=100000)
    # Parameters for evaluator
    parser.add_argument('--evaluator_name', type=str, default='evaluator')
    parser.add_argument('--num_eval_episode', type=int, default=10)
    # Data savings
    parser.add_argument('--save_folder', type=str, default=None)
    parser.add_argument('--apprfunc_save_interval', type=int, default=1000)
    parser.add_argument('--log_save_interval', type=int, default=50)  # reward?

    # get parameter dict
    args = vars(parser.parse_args())
    env = create_env(**args)
    args['obsv_dim'] = env.observation_space.shape[0]
    args['action_num'] = env.action_space.n
    if args['noise_params'] is None:
        args['noise_params'] = {
            'action_num': args['action_num'],
            'epsilon': 0.1  # TODO: make configurable
        }

    # create save arguments
    if args['save_folder'] is None:
        args['save_folder'] = os.path.join('../results/' + parser.parse_args().algorithm, datetime.datetime.now().strftime("%m-%d-%H-%M"))
    os.makedirs(args['save_folder'], exist_ok=True)
    os.makedirs(args['save_folder'] + '/apprfunc', exist_ok=True)
    with open(args['save_folder'] + '/config.json', 'w', encoding='utf-8') as f:
        json.dump(change_type(copy.deepcopy(args)), f, ensure_ascii=False, indent=4)

    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    # start training
    trainer.train()

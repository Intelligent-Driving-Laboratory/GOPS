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
from modules.create_pkg.create_env import create_env
from modules.create_pkg.create_alg import create_alg
from modules.create_pkg.create_trainer import create_trainer


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    # Key Parameters for users
    parser.add_argument('--env_id', type=str, default='gym_cartpole', help='')
    parser.add_argument('--apprfunc', type=str, default='MLP', help='')
    parser.add_argument('--algorithm', type=str, default='DQN', help='')
    parser.add_argument('--trainer', type=str, default='serial_trainer', help='')
    parser.add_argument('--savefile', type=str, default=None, help='')

    # 1. Parameters for environment
    parser.add_argument('--obsv_dim', type=int, default=None, help='')
    parser.add_argument('--action_type', type=str, default='disc', choices=['disc', 'conti'],help='Action space type: discrete or continuous')
    parser.add_argument('--action_dim', type=int, default=1,help='')
    parser.add_argument('--action_num', type=int, default=None,help='Num of discrete actions (similar to gym.spaces.Discrete)')
    parser.add_argument('--max_episode_length',type=int, default=200, help='')

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
    parser.add_argument('--delay_update', type=int, default=1, help='')
    parser.add_argument('--max_sampled_number', type=int, default=2000)
 
    # 4. Parameters for trainer
    parser.add_argument('--max_train_episode', type=int, default=2000, help='')
    parser.add_argument('--episode_length', type=int, default=200, help='')
    parser.add_argument('--max_sample_num', type=int, default=100000, help='')

    parser.add_argument('--eval_length', type=int, default=parser.parse_args().episode_length, help='')
    parser.add_argument('--buffer_name', type=str, default='replay_buffer')
    parser.add_argument('--buffer_warm_size', type=int, default=1000)
    parser.add_argument('--buffer_max_size', type=int, default=100000)
    parser.add_argument('--noise', type=float, default=0.2, help='')
    parser.add_argument('--reward_scale', type=float, default=0.1, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--is_render', type=bool, default=True)

    # Data savings
    parser.add_argument('--save_folder', type=str,default='./results/' + parser.parse_args().algorithm)
    parser.add_argument('--apprfunc_save_interval', type=int, default=1000)
    parser.add_argument('--log_save_interval', type=int, default=10) # reward?


    # get parameter dict
    args = vars(parser.parse_args())

    # Step 1: create environment
    env = create_env(**args)  #
    args['obsv_dim'] = env.observation_space.shape[0]
    args['action_num'] = env.action_space.n
    
    # Step 2: create algorithm and approximate function
    alg = create_alg(**args) # create appr_model in algo **vars(args)

    # Step 3: create trainer
    trainer = create_trainer(env, alg, **args)

    # start training
    trainer.train()
    print("Training is Done!")

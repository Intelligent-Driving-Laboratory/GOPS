#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yang Yujie

import argparse
import os
import numpy as np

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot import plot_all
from gops.utils.tensorboard_tools import start_tensorboard, save_tb_to_csv

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument('--env_id', type=str, default='gym_pendulum')
    parser.add_argument('--algorithm', type=str, default='PPO')
    parser.add_argument('--enable_cuda', default=False, help='Disable CUDA')

    ################################################
    # 1. Parameters for environment
    parser.add_argument('--obsv_dim', type=int, default=None) # dim(State)
    parser.add_argument('--action_dim', type=int, default=None) # dim(Action)
    parser.add_argument('--action_high_limit', type=list, default=None)
    parser.add_argument('--action_low_limit', type=list, default=None)
    parser.add_argument('--action_type', type=str, default='continu') # Options: continu/discret
    parser.add_argument('--is_render', type=bool, default=False) # Draw environment animation
    parser.add_argument('--is_adversary', type=bool, default=False, help='Adversary training')

    ################################################
    # 2.1 Parameters of value approximate function
    # Options: StateValue/ActionValue/ActionValueDis
    parser.add_argument('--value_func_name', type=str, default='StateValue')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--value_func_type', type=str, default='MLP')
    value_func_type = parser.parse_args().value_func_type
    # 2.1.1 MLP, CNN, RNN
    parser.add_argument('--value_hidden_sizes', type=list, default=[64, 64])
    # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
    parser.add_argument('--value_hidden_activation', type=str, default='relu')
    # Output Layer: linear
    parser.add_argument('--value_output_activation', type=str, default='linear')

    # 2.2 Parameters of policy approximate function
    # Options: None/DetermPolicy/StochaPolicy
    parser.add_argument('--policy_func_name', type=str, default='StochaPolicy')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--policy_func_type', type=str, default='POLY')
    parser.add_argument('--policy_act_distribution', type=str, default='default')
    policy_func_type = parser.parse_args().policy_func_type
    ### 2.2.1 MLP, CNN, RNN
    if policy_func_type == 'POLY':
        pass
    parser.add_argument('--policy_min_log_std', type=int, default=-3)
    parser.add_argument('--policy_max_log_std', type=int, default=4)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='3e-4 in the paper')

    ################################################
    # 4. Parameters for trainer
    # Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer
    parser.add_argument('--trainer', type=str, default='on_serial_trainer')
    # Maximum iteration number
    parser.add_argument('--max_iteration', type=int, default=300)
    trainer_type = parser.parse_args().trainer
    parser.add_argument('--ini_network_dir', type=str, default=None)
    # 4.1. Parameters for on_serial_trainer
    parser.add_argument('--num_repeat', type=int, default=20, help='5')  # 5 repeat
    parser.add_argument('--num_mini_batch', type=int, default=8, help='8')  # 8 mini_batch
    parser.add_argument('--mini_batch_size', type=int, default=128, help='128')  # 8 mini_batch * 128 = 1024
    parser.add_argument('--num_epoch', type=int,
                        default=parser.parse_args().num_repeat * parser.parse_args().num_mini_batch,
                        help='# 50 gradient step per sample')
    ################################################
    # 5. Parameters for sampler
    parser.add_argument('--sampler_name', type=str, default='on_sampler')
    # Batch size of sampler for buffer store
    parser.add_argument('--sample_batch_size', type=int, default=1024,
                        help='Batch size of sampler for buffer store = 1024')  # 8 env * 128 step
    assert parser.parse_args().num_mini_batch * parser.parse_args().mini_batch_size == parser.parse_args().sample_batch_size, 'sample_batch_size error'
    # Add noise to actions for better exploration
    parser.add_argument('--noise_params', type=dict,
                        default=None,
                        help='Add noise to actions for exploration')
    ################################################
    # 6. Parameters for buffer
    parser.add_argument('--buffer_name', type=str, default='replay_buffer')
    parser.add_argument('--buffer_warm_size', type=int, default=1000)
    parser.add_argument('--buffer_max_size', type=int, default=100000)

    ################################################
    # 7. Parameters for evaluator
    parser.add_argument('--evaluator_name', type=str, default='evaluator')
    parser.add_argument('--num_eval_episode', type=int, default=5)
    parser.add_argument('--eval_interval', type=int, default=1)

    ################################################
    # 8. Data savings
    parser.add_argument('--save_folder', type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument('--apprfunc_save_interval', type=int, default=20,
                        help='Save value/policy every N updates')
    # Save key info every N updates
    parser.add_argument('--log_save_interval', type=int, default=1,
                        help='Save gradient time/critic loss/actor loss/average value every N updates')

    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args['save_folder'])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    alg.set_parameters({'gamma': 0.99, 'loss_coefficient_value': 0.5, 'loss_coefficient_entropy': 0.01,
                        'schedule_adam':'None','schedule_clip':'linear','loss_value_clip':False,
                       'loss_value_norm':True})
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    # Start training ... ...
    trainer.train()
    print('Training is finished!')

    # Plot and save training figures
    plot_all(args['save_folder'])
    save_tb_to_csv(args['save_folder'])

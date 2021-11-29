#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Jie Li
#  Description: PPO algorithm, MLP, continuous version of cart pole, on serial trainer
#  Update Date: 2021-06-11


#  General Optimal control Problem Solver (GOPS)


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
from modules.utils.init_args import init_args
from modules.utils.plot import plot_all
from modules.utils.tensorboard_tools import start_tensorboard, save_tb_to_csv

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument('--env_id', type=str, default='gym_cartpoleconti', help='')
    parser.add_argument('--algorithm', type=str, default='PPO', help='')
    parser.add_argument('--enable_cuda', default=False, help='Disable CUDA')

    ################################################
    # 1. Parameters for environment
    parser.add_argument('--obsv_dim', type=int, default=None, help='')
    parser.add_argument('--action_dim', type=int, default=None, help='')
    parser.add_argument('--action_high_limit', type=list, default=None, help='')
    parser.add_argument('--action_low_limit', type=list, default=None, help='')
    parser.add_argument('--action_type', type=str, default='continu', help='')
    parser.add_argument('--is_render', type=bool, default=False, help='')
    parser.add_argument('--is_adversary', type=bool, default=False, help='Adversary training')
    parser.add_argument('--is_constrained', type=bool, default=False, help='Constrained training')

    ################################################
    # 2.1 Parameters of value approximate function
    # Options: StateValue/ActionValue/ActionValueDis
    parser.add_argument('--value_func_name', type=str, default='StateValue')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--value_func_type', type=str, default='MLP')
    value_func_type = parser.parse_args().value_func_type
    # 2.1.1 MLP, CNN, RNN
    parser.add_argument('--value_hidden_sizes', type=list, default=[64, 64, 64])
    # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
    parser.add_argument('--value_hidden_activation', type=str, default='relu')
    # Output Layer: linear
    parser.add_argument('--value_output_activation', type=str, default='linear')

    # 2.2 Parameters of policy approximate function
    # Options: None/DetermPolicy/StochaPolicy
    parser.add_argument('--policy_func_name', type=str, default='StochaPolicy')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--policy_func_type', type=str, default='MLP')
    policy_func_type = parser.parse_args().policy_func_type
    # 2.2.1 MLP, CNN, RNN
    parser.add_argument('--policy_hidden_sizes', type=list, default=[64, 64])
    # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
    parser.add_argument('--policy_hidden_activation', type=str, default='relu')
    # Output Layer: linear
    parser.add_argument('--policy_output_activation', type=str, default='linear')
    parser.add_argument('--policy_min_log_std', type=int, default=-8)  # -6
    parser.add_argument('--policy_max_log_std', type=int, default=2)  # 3

    ################################################
    # 3. Parameters for algorithm
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='3e-4 in the paper')

    ################################################
    # 4. Parameters for trainer
    # Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer
    parser.add_argument('--trainer', type=str, default='on_serial_trainer')
    # Maximum iteration number
    parser.add_argument('--max_iteration', type=int, default=2400)
    trainer_type = parser.parse_args().trainer
    parser.add_argument('--ini_network_dir', type=str, default=None)
    # 4.1. Parameters for on_serial_trainer
    parser.add_argument('--num_repeat', type=int, default=10, help='5')  # 5 repeat
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
                        default={'mean': np.array([0], dtype=np.float32),
                                 'std': np.array([1e-6], dtype=np.float32)},
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
    parser.add_argument('--eval_interval', type=int, default=3*parser.parse_args().num_epoch)
    parser.add_argument('--print_interval', type=int, default=3*parser.parse_args().num_epoch)

    ################################################
    # 8. Data savings
    parser.add_argument('--save_folder', type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument('--apprfunc_save_interval', type=int, default=3*parser.parse_args().num_epoch,
                        help='Save value/policy every N updates')
    # Save key info every N updates
    parser.add_argument('--log_save_interval', type=int, default=3*parser.parse_args().num_epoch,
                        help='Save gradient time/critic loss/actor loss/average value every N updates')

    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args['save_folder'])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    alg.set_parameters({'gamma': 0.995, 'loss_coefficient_value': 0.5, 'loss_coefficient_entropy': 0.01})
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

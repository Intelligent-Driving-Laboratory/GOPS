#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: continuous version of Cartpole Enviroment
#  Update Date: 2021-07-11, Yuxuan Jiang & Guojian Zhan : TRPO with cartpoleconti


import argparse

import numpy as np
import multiprocessing
from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot import plot_all
from gops.utils.tensorboard_tools import start_tensorboard, save_tb_to_csv

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument('--env_id', type=str, default='gym_lunarlanderconti')
    parser.add_argument('--algorithm', type=str, default='TRPO')
    parser.add_argument('--enable_cuda', default=False, help='Enable CUDA')

    ################################################
    # 1. Parameters for environment
    parser.add_argument('--obsv_dim', type=int, default=None)  # dim(State)
    parser.add_argument('--action_dim', type=int, default=None)  # dim(Action)
    parser.add_argument('--action_high_limit', type=list, default=None)
    parser.add_argument('--action_low_limit', type=list, default=None)
    parser.add_argument('--action_type', type=str, default='continu')  # Options: continu/discret
    parser.add_argument('--is_render', type=bool, default=True)  # Draw environment animation
    parser.add_argument('--is_adversary', type=bool, default=False)

    ################################################
    # 2.1 Parameters of value approximate function
    # Options: StateValue/ActionValue/ActionValueDis
    parser.add_argument('--value_func_name', type=str, default='StateValue')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--value_func_type', type=str, default='MLP')
    value_func_type = parser.parse_args().value_func_type
    ### 2.1.1 MLP, CNN, RNN
    if value_func_type == 'MLP':
        parser.add_argument('--value_hidden_sizes', type=list, default=[256, 128])
        # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
        parser.add_argument('--value_hidden_activation', type=str, default='relu')
        # Output Layer: linear
        parser.add_argument('--value_output_activation', type=str, default='linear')

    # 2.2 Parameters of policy approximate function
    # Options: None/DetermPolicy/StochaPolicy
    parser.add_argument('--policy_func_name', type=str, default='StochaPolicy')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--policy_func_type', type=str, default='MLP')
    parser.add_argument('--policy_act_distribution', type=str, default='default')
    policy_func_type = parser.parse_args().policy_func_type
    ### 2.2.1 MLP, CNN, RNN
    if policy_func_type == 'MLP':
        parser.add_argument('--policy_hidden_sizes', type=list, default=[256, 128])
        # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
        parser.add_argument('--policy_hidden_activation', type=str, default='relu')
        # Output Layer: tanh
        parser.add_argument('--policy_output_activation', type=str, default='linear')  # already tanh-ed in mlp

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument('--value_learning_rate', type=float, default=1e-4)

    parser.add_argument('--delta', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.97)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--atol', type=float, default=1e-8)
    parser.add_argument('--damping_factor', type=float, default=0.1)
    parser.add_argument('--max_cg', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--max_search', type=int, default=10)
    parser.add_argument('--train_v_iters', type=int, default=40)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument('--trainer', type=str, default='on_sync_trainer')
    # Maximum iteration number
    parser.add_argument('--max_iteration', type=int, default=5000)
    trainer_type = parser.parse_args().trainer
    parser.add_argument('--ini_network_dir', type=str, default=None)
    # 4.3. Parameters for sync trainer
    if trainer_type == 'on_sync_trainer':
        parser.add_argument('--num_epoch', type=int,
                            default=1,
                            help='# 50 gradient step per sample')
        import ray

        ray.init()
        parser.add_argument('--num_samplers', type=int, default=2, help='number of samplers')
        cpu_core_num = multiprocessing.cpu_count()
        num_core_input = parser.parse_args().num_samplers + 3
        if num_core_input > cpu_core_num:
            raise ValueError('The number of core is {}, but you want {}!'.format(cpu_core_num, num_core_input))

    ################################################
    # 5. Parameters for sampler
    parser.add_argument('--sampler_name', type=str, default='on_sampler')
    # Batch size of sampler for buffer store
    parser.add_argument('--sample_batch_size', type=int, default=1024)
    # Add noise to actions for better exploration
    parser.add_argument('--noise_params', type=dict,
                        default={'mean': np.array([0.], dtype=np.float32),
                                 'std': np.array([0.], dtype=np.float32)})

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument('--evaluator_name', type=str, default='evaluator')
    parser.add_argument('--num_eval_episode', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=10)

    ################################################
    # 8. Data savings
    parser.add_argument('--save_folder', type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument('--apprfunc_save_interval', type=int, default=500)
    # Save key info every N updates
    parser.add_argument('--log_save_interval', type=int, default=10)

    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args['save_folder'])
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
    # Start training ... ...
    trainer.train()

    print('Training is finished!')

    # Plot and save training figures
    plot_all(args['save_folder'])
    save_tb_to_csv(args['save_folder'])
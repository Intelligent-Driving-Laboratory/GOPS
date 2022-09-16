#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Soft Actor Critic
#  Update Date: 2021-06-11, Yang Yujie: create SAC algorithm


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
    parser.add_argument("--env_id", type=str, default="pyth_inverteddoublependulum")
    parser.add_argument("--algorithm", type=str, default="SAC")
    parser.add_argument("--enable_cuda", default=False, help="Disable CUDA")

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--obsv_dim", type=int, default=None)  # dim(State)
    parser.add_argument("--action_dim", type=int, default=None)  # dim(Action)
    parser.add_argument("--action_high_limit", type=list, default=None)
    parser.add_argument("--action_low_limit", type=list, default=None)
    parser.add_argument(
        "--action_type", type=str, default="continu"
    )  # Options: continu/discret
    parser.add_argument(
        "--is_render", type=bool, default=False
    )  # Draw environment animation
    parser.add_argument(
        "--is_adversary", type=bool, default=False, help="Adversary training"
    )

    ################################################
    # 2.1 Parameters of value approximate function
    # Options: StateValue/ActionValue/ActionValueDis
    parser.add_argument("--value_func_name", type=str, default="ActionValue")
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument("--value_func_type", type=str, default="MLP")
    value_func_type = parser.parse_known_args()[0].value_func_type
    ### 2.1.1 MLP, CNN, RNN
    if value_func_type == "MLP":
        parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256, 256])
        # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
        parser.add_argument("--value_hidden_activation", type=str, default="relu")
        # Output Layer: linear
        parser.add_argument("--value_output_activation", type=str, default="linear")

    # 2.2 Parameters of policy approximate function
    # Options: None/DetermPolicy/StochaPolicy
    parser.add_argument("--policy_func_name", type=str, default="StochaPolicy")
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument(
        "--policy_act_distribution", type=str, default="TanhGaussDistribution"
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    ### 2.2.1 MLP, CNN, RNN
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256, 256])
        # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
        parser.add_argument("--policy_hidden_activation", type=str, default="relu")
        # Output Layer: tanh
        parser.add_argument("--policy_output_activation", type=str, default="linear")
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=1)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=3e-4)
    parser.add_argument("--q_learning_rate", type=float, default=3e-4)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)
    parser.add_argument("--alpha_learning_rate", type=float, default=5e-5)

    ################################################
    # 4. Parameters for trainer
    # Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=50_000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--ini_network_dir", type=str, default=None)
    # 4.3. Parameters for off_serial_trainer
    if trainer_type == "off_serial_trainer":
        parser.add_argument("--buffer_name", type=str, default="replay_buffer")
        # Size of collected samples before training
        parser.add_argument("--buffer_warm_size", type=int, default=1000)
        # Max size of reply buffer
        parser.add_argument("--buffer_max_size", type=int, default=int(1e6))
        # Batch size of replay samples from buffer
        parser.add_argument("--replay_batch_size", type=int, default=256)
        # Period of sync central policy of each sampler
        parser.add_argument("--sampler_sync_interval", type=int, default=1)
    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument("--sample_interval", type=int, default=1)
    # Add noise to actions for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 7. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)

    ################################################
    # 8. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=3000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=100)

    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    alg.set_parameters({"reward_scale": 0.1, "gamma": 0.99, "tau": 0.005})
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
    print("Training is finished!")

    # Plot and save training figures
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Separated Proportional-Integral Lagrangian Algorithm
#  Paper: https://ieeexplore.ieee.org/document/9785377
#  Update: 2021-03-05, Baiyu Peng: create SPIL algorithm


import argparse
import os
import numpy as np
import multiprocessing

# import sys
# gops_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
# sys.path.insert(0, gops_path)

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_mobilerobot")
    parser.add_argument("--algorithm", type=str, default="SPIL")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")

    # 1. Parameters for environment
    parser.add_argument("--obsv_dim", type=int, default=None)
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--constrained_dim", type=int, default=None)
    parser.add_argument("--action_high_limit", type=list, default=None)
    parser.add_argument("--action_low_limit", type=list, default=None)
    parser.add_argument("--action_type", type=str, default="continu")
    parser.add_argument("--is_render", type=bool, default=False)
    parser.add_argument("--is_adversary", type=bool, default=False)

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument("--value_func_name", type=str, default="StateValue")
    parser.add_argument("--value_func_type", type=str, default="MLP")
    value_func_type = parser.parse_known_args()[0].value_func_type
    if value_func_type == "MLP":
        parser.add_argument("--value_hidden_sizes", type=list, default=[64, 64])
        parser.add_argument("--value_hidden_activation", type=str, default="relu")
        parser.add_argument("--value_output_activation", type=str, default="linear")
    # 2.2 Parameters of policy approximate function
    parser.add_argument("--policy_func_name", type=str, default="DetermPolicy")
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[64, 64])
        parser.add_argument(
            "--policy_hidden_activation", type=str, default="relu", help=""
        )
        parser.add_argument(
            "--policy_output_activation", type=str, default="tanh", help=""
        )

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=2e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=0.3e-3)

    # 4. Parameters for trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    parser.add_argument(
        "--max_iteration", type=int, default=10000, help="Maximum iteration number"
    )
    parser.add_argument("--ini_network_dir", type=str, default=None)
    trainer_type = parser.parse_known_args()[0].trainer
    if trainer_type == "off_serial_trainer":
        parser.add_argument("--buffer_name", type=str, default="replay_buffer")
        parser.add_argument("--buffer_warm_size", type=int, default=1000)
        parser.add_argument("--buffer_max_size", type=int, default=400 * 1000)
        parser.add_argument("--replay_batch_size", type=int, default=1024)
        parser.add_argument("--sampler_sync_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    parser.add_argument("--sample_batch_size", type=int, default=256)
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={
            "mean": np.array([0, 0], dtype=np.float32),
            "std": np.array([0.05, 0.05], dtype=np.float32),
        },
    )

    ################################################
    # 7. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=100)

    ################################################
    # 8. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--apprfunc_save_interval", type=int, default=100)
    parser.add_argument("--log_save_interval", type=int, default=100)

    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)
    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)  # create appr_model in algo **vars(args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)  # 调用alg里面的函数，创建自己的网络
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    # plot and save training curve
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])

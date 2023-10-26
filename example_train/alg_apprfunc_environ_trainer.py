#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: template for training examples
#  Update: 2020-11-10, Hao Sun: create example template
#  Update: 2021-05-21, Shengbo Eben Li: reformulate code formats
#  Update: 2022-12-03, Wenxuan Wang: update example template
#  Update: 2022-12-10, Shengbo Li: minor revision

import argparse
import os
from typing import Any

import numpy as np

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
    parser.add_argument("--env_id", type=str, default="gym_cartpoleconti", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="DDPG", help="name of RL/ADP algorithm")
    parser.add_argument("--enable_cuda", default=False, help="enable CUDA")

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--reward_scale", type=float, default=1, help="reward scale factor")
    parser.add_argument("--reward_shift", type=float, default=0, help="reward shift factor")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw training animation")
    parser.add_argument("--seed", default=3328005365, help="seeds of env & network initialization")

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValue",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_known_args()[0].value_func_type
    # 2.1.1 MLP/RNN
    if value_func_type == "MLP" or value_func_type == "RNN":
        parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256, 128])
        parser.add_argument(
            "--value_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
        )
        parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")
    # 2.1.2 CNN/CNN_SHARED
    if value_func_type == "CNN" or value_func_type == "CNN_SHARED":
        parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256, 128])
        parser.add_argument(
            "--value_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
        )
        parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")
        parser.add_argument("--value_conv_type", type=str, default="type_2", help="Options: type_1/type_2")
    # 2.1.3 Polynomial
    if value_func_type == "POLY":
        parser.add_argument("--value_degree", type=int, default=2, help="degree of poly function")
        parser.add_argument("--value_add_bias", type=bool, default=False, help="add 0 degree term")
    # 2.1.4 Gauss Radical Func
    if value_func_type == "GAUSS":
        parser.add_argument("--value_num_kernel", type=int, default=30, help="num of kernels")

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="DetermPolicy",
        help="Options: None/DetermPolicy/StochaPolicy/FiniteHorizonPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    # 2.2.1 MLP/RNN
    if policy_func_type == "MLP" or policy_func_type == "RNN":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256, 128])
        parser.add_argument(
            "--policy_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
        )
    # 2.2.2 CNN/CNN_SHARED
    if policy_func_type == "CNN" or policy_func_type == "CNN_SHARED":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256, 128])
        parser.add_argument(
            "--policy_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
        )
        parser.add_argument("--policy_conv_type", type=str, default="type_2", help="Options: type_1/type_2")
    # 2.2.3 Polynomial
    if policy_func_type == "POLY":
        parser.add_argument("--policy_degree", type=int, default=2, help="degree of poly function")
        parser.add_argument("--policy_add_bias", type=bool, default=False, help="add zero-order term")
    # 2.2.4 Gauss Radical Func
    if policy_func_type == "GAUSS":
        parser.add_argument("--policy_num_kernel", type=int, default=35, help="num of kernels")

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=1e-4)
    parser.add_argument("--policy_learning_rate", type=float, default=1e-5)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=5000)
    parser.add_argument("--ini_network_dir",
                        type=str, default=None,
                        help="Path of initial networks")
    trainer_type = parser.parse_known_args()[0].trainer

    # 4.1. Parameters for on-policy serial trainer
    if trainer_type == "on_serial_trainer":
        pass
    # 4.2. Parameters for on-policy sync trainer
    if trainer_type == "on_sync_trainer":
        pass
    # 4.3. Parameters for off-policy serial trainer
    if trainer_type == "off_serial_trainer":
        parser.add_argument(
            "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
        )
        # Size of collected samples before training, i.e., warm-start
        parser.add_argument("--buffer_warm_size", type=int, default=1000) # count as samples
        # Max size of reply buffer
        parser.add_argument("--buffer_max_size", type=int, default=100000) # count as samples
        # Batch size of replay samples from buffer
        parser.add_argument("--replay_batch_size", type=int, default=1024) # count as samples
        # Period of sampling
        parser.add_argument("--sample_interval", type=int, default=1)
    # 4.4. Parameters for off-policy async trainer
    if trainer_type == "off_async_trainer":
        import ray
        ray.init()
        parser.add_argument("--num_algs", type=int, default=2)
        parser.add_argument("--num_samplers", type=int, default=2)
        parser.add_argument("--num_buffers", type=int, default=1)
        # Note that num of algs+samplers+buffers <= num of computer cores
        cpu_core_num = multiprocessing.cpu_count()
        num_core_input = (
                parser.parse_known_args()[0].num_algs
                + parser.parse_known_args()[0].num_samplers
                + parser.parse_known_args()[0].num_buffers
                + 2
        )
        if num_core_input > cpu_core_num:
            raise ValueError("The number of parallel cores is too large!")
        parser.add_argument("--buffer_name", type=str, default="replay_buffer")
        parser.add_argument("--buffer_warm_size", type=int, default=1000)  # count as samples
        parser.add_argument("--buffer_max_size", type=int, default=100000) # count as samples
        parser.add_argument("--replay_batch_size", type=int, default=1024)  # count as samples
        parser.add_argument("--sample_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="on_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=256)
    # Add noise to action for better exploration
    parser.add_argument(
        "--noise_params",
        type=Any,
        default={"mean": np.array([0], dtype=np.float32), "std": np.array([0.1], dtype=np.float32),},
        help="Only used for continuous action space",
    )

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator", help="name of policy evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=5, help="num of episodes in each evaluation")
    parser.add_argument("--eval_interval", type=int, default=200, help="iteration interval to perform each evaluation")
    parser.add_argument("--eval_save", type=str, default=True, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=500, help="interval to save data in folder")
    # Display key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=200, help="interval to display in tensorboard")

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args["save_folder"])
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

    ################################################
    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    ################################################
    # Plot and save training figures
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")

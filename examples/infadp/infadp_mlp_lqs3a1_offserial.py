#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Infinite ADP algorithm in continute version of Cartpole Enviroment
#  Update Date: 2020-11-10, Wenxuan Wang: adjust parameters
#  Update Date: 2022-10-24, Xujie Song  : ensure control precison


import argparse
import os

os.environ["OMP_NUM_THREADS"] = "4"
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


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_lq")
    parser.add_argument("--lq_config", type=str, default="s3a1")
    parser.add_argument("--algorithm", type=str, default="INFADP")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")

    # 1. Parameters for environment
    parser.add_argument("--obsv_dim", type=int, default=None)
    parser.add_argument("--action_dim", type=int, default=None)
    parser.add_argument("--action_high_limit", type=list, default=None)
    parser.add_argument("--action_low_limit", type=list, default=None)
    parser.add_argument("--action_type", type=str, default="continu")
    parser.add_argument("--reward_scale", type=float, default=1)
    parser.add_argument("--reward_shift", type=float, default=0)
    parser.add_argument("--is_render", type=bool, default=False)
    parser.add_argument(
        "--is_adversary", type=bool, default=False, help="Adversary training"
    )
    parser.add_argument(
        "--is_constrained", type=bool, default=False, help="Adversary training"
    )
    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument("--value_func_name", type=str, default="StateValue")
    parser.add_argument("--value_func_type", type=str, default="MLP")
    value_func_type = parser.parse_known_args()[0].value_func_type
    if value_func_type == "MLP":
        parser.add_argument("--value_hidden_sizes", type=list, default=[64, 64])
        parser.add_argument("--value_hidden_activation", type=str, default="gelu")
        parser.add_argument("--value_output_activation", type=str, default="linear")
    # 2.2 Parameters of policy approximate function
    parser.add_argument("--policy_func_name", type=str, default="DetermPolicy")
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[64, 64])
        parser.add_argument(
            "--policy_hidden_activation", type=str, default="gelu", help=""
        )
        parser.add_argument(
            "--policy_output_activation", type=str, default="linear", help=""
        )

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=8e-4)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)

    # 4. Parameters for trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    parser.add_argument(
        "--max_iteration", type=int, default=20000, help="Maximum iteration number"
    )
    parser.add_argument("--ini_network_dir", type=str, default=None)
    trainer_type = parser.parse_known_args()[0].trainer
    if trainer_type == "off_serial_trainer":
        parser.add_argument("--buffer_name", type=str, default="replay_buffer")
        parser.add_argument("--buffer_warm_size", type=int, default=1000)
        parser.add_argument("--buffer_max_size", type=int, default=100000)
        parser.add_argument("--replay_batch_size", type=int, default=64)
        parser.add_argument("--sampler_sync_interval", type=int, default=1)
    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    parser.add_argument("--sample_batch_size", type=int, default=8)
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={
            "mean": np.array([0], dtype=np.float32),
            "std": np.array([0.0], dtype=np.float32),
        },
    )

    ################################################
    # 7. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=100)
    # set train_space & work_space
    parser.add_argument("--initial_distribution", type=str, default="uniform")
    init_mean = np.array([0, 0, 0], dtype=np.float32)
    init_std = np.array([2, 2, 2], dtype=np.float32)
    train_space = np.stack((init_mean - 1 * init_std, init_mean + 1 * init_std))
    work_space = np.stack((init_mean - 0.5 * init_std, init_mean + 0.5 * init_std))
    parser.add_argument("--train_space", type=np.array, default=train_space)
    parser.add_argument("--work_space", type=np.array, default=work_space)

    ################################################
    # 8. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--apprfunc_save_interval", type=int, default=500)
    parser.add_argument("--log_save_interval", type=int, default=100)

    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)
    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)  # create appr_model in algo **vars(args)
    alg.set_parameters({"gamma": 0.99, "tau": 0.2,"forward_step":200})
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

    # plot and save training curve
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])

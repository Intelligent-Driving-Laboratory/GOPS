#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for fhadp + veh3dofconti + mlp + off_serial
#  Update Date: 2023-08-28, Guojian Zhan: support lr schedule

import os
import argparse
import json
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
from gops.utils.common_utils import seed_everything

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="quadrotor_1dof_tracking_stablization")
    parser.add_argument("--algorithm", type=str, default="FHADP")
    parser.add_argument("--pre_horizon", type=int, default=30)
    parser.add_argument("--enable_cuda", default=False)
    parser.add_argument("--seed", default=None, help="seed")
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--is_render", type=bool, default=True)
    parser.add_argument("--is_adversary", type=bool, default=False)

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument("--value_func_type", type=str, default="MLP")

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="FiniteHorizonPolicy"
    )
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument("--policy_hidden_activation", type=str, default="elu")
    
    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--policy_learning_rate", type=float, default=1e-3)
    parser.add_argument("--policy_scheduler", type=json.loads, default={
        "name": "LinearLR",
        "params": {
            "start_factor": 1.0,
            "end_factor": 0.0,
            "total_iters": 100000,
            }
    })

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer")
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=2000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    # 4.1. Parameters for off_serial_trainer
    parser.add_argument("--buffer_name", type=str, default="replay_buffer")
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=1000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=100000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=64)
    # Period of sampling
    parser.add_argument("--sample_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=256)
    # Add noise to action for better exploration
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={"mean": np.array([0], dtype=np.float32), "std": np.array([0.2], dtype=np.float32),},
    )

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_save", type=str, default=True, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=100)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=100)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)
    # start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    # buffer = create_buffer(**args)
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

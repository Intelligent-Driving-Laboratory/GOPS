#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: veh3dofconti_model tracking
#  Update Date: 2022-04-29, Jiaxin Gao: create example

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse
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
from gops.utils.utils import seed_everything

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="pyth_veh2dofconti")
    parser.add_argument("--algorithm", type=str, default="FHADP")
    parser.add_argument("--pre_horizon", type=int, default=30)
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--obsv_dim", type=int, default=4, help="dim(State)")
    parser.add_argument("--action_dim", type=int, default=1, help="dim(Action)")
    parser.add_argument("--action_high_limit", type=list, default=None)
    parser.add_argument("--action_low_limit", type=list, default=None)
    parser.add_argument(
        "--action_type", type=str, default="continu", help="Options: continu/discret"
    )
    parser.add_argument(
        "--is_render", type=bool, default=False, help="Draw environment animation"
    )
    parser.add_argument(
        "--is_adversary", type=bool, default=False, help="Adversary training"
    )
    ################################################
    # 2.1 Parameters of value approximate function
    # parser.add_argument("--value_func_name", type=str, default="ActionValue")
    parser.add_argument("--value_func_type", type=str, default="MLP")

    # 2.2 Parameters of policy approximate function
    parser.add_argument("--policy_func_name", type=str, default="DetermPolicy")
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
        parser.add_argument("--policy_hidden_activation", type=str, default="elu")
        parser.add_argument("--policy_output_activation", type=str, default="linear")

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--policy_learning_rate", type=float, default=3e-5)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument("--trainer", type=str, default="on_serial_trainer")
    parser.add_argument("--max_iteration", type=int, default=1000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--ini_network_dir", type=str, default=None)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="on_sampler")
    parser.add_argument("--sample_batch_size", type=int, default=200)
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={
            "mean": np.array([0], dtype=np.float32),
            "std": np.array([0.2], dtype=np.float32),
        },
    )

    ################################################
    # 7. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=100)
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
    # start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    alg.set_parameters(
        {"reward_scale": 0.1, "gamma": 0.99, "tau": 0.2, "delay_update": 1}
    )
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = None
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    # Plot and save training figures
    # plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Discrete version of Cartpole Enviroment
#  Update Date: 2020-11-10, Hao Sun: renew env para
#  Update Date: 2021-05-21, Shengbo Li: Reformualte code formats
#  Update Date: 2021-06-01, Shengbo Li: General Setup for GOPS examples


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
    parser.add_argument("--env_id", type=str, default="gym_cartpoleconti")
    parser.add_argument("--algorithm", type=str, default="DDPG")

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

    ################################################
    # 2.1 Parameters of value approximate function
    # Options: StateValue/ActionValue/ActionValueDis
    parser.add_argument("--value_func_name", type=str, default="ActionValue")
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument("--value_func_type", type=str, default="MLP")
    value_func_type = parser.parse_known_args()[0].value_func_type
    ### 2.1.1 MLP, CNN, RNN
    if value_func_type == "MLP":
        parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256, 128])
        # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
        parser.add_argument("--value_hidden_activation", type=str, default="relu")
        # Output Layer: linear
        parser.add_argument("--value_output_activation", type=str, default="linear")
    ### 2.1.2 Polynominal
    if value_func_type == "POLY":
        pass
    ### 2.1.3 Gauss Radical Func
    if value_func_type == "GAUSS":
        parser.add_argument("--value_num_kernel", type=int, default=30)

    # 2.2 Parameters of policy approximate function
    # Options: None/DetermPolicy/StochaPolicy
    parser.add_argument("--policy_func_name", type=str, default="DetermPolicy")
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    ### 2.2.1 MLP, CNN, RNN
    if policy_func_type == "MLP":
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
        # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
        parser.add_argument("--policy_hidden_activation", type=str, default="relu")
        # Output Layer: tanh
        parser.add_argument("--policy_output_activation", type=str, default="tanh")
    ### 2.2.2 Polynominal
    if policy_func_type == "POLY":
        pass
    ### 2.2.3 Gauss Radical Func
    if policy_func_type == "GAUSS":
        parser.add_argument("--policy_num_kernel", type=int, default=35)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--value_learning_rate", type=float, default=1e-4)
    parser.add_argument("--policy_learning_rate", type=float, default=1e-5)
    parser.add_argument("--delay_update", type=int, default=1, help="")
    # Reward = reward_scale * environment.Reward
    parser.add_argument("--reward_scale", type=float, default=1)

    ################################################
    # 4. Parameters for trainer
    # Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=5000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--ini_network_dir", type=str, default=None)
    # 4.1. Parameters for on_serial_trainer
    if trainer_type == "on_serial_trainer":
        pass
    # 4.2. Parameters for on_sync_trainer
    if trainer_type == "on_sync_trainer":
        pass
    # 4.3. Parameters for off_serial_trainer
    if trainer_type == "off_serial_trainer":
        parser.add_argument("--buffer_name", type=str, default="replay_buffer")
        # Size of collected samples before training
        parser.add_argument("--buffer_warm_size", type=int, default=1000)
        # Max size of reply buffer
        parser.add_argument("--buffer_max_size", type=int, default=1e5)
        # Batch size of replay samples from buffer
        parser.add_argument("--replay_batch_size", type=int, default=1024)
        # Period of sync central policy of each sampler
        parser.add_argument("--sampler_sync_interval", type=int, default=1)
    # 4.4. Parameters for off_async_trainer
    if trainer_type == "off_async_trainer":
        import ray

        ray.init()
        parser.add_argument("--num_algs", type=int, default=2)
        parser.add_argument("--num_samplers", type=int, default=2)
        parser.add_argument("--num_buffers", type=int, default=1)
        parser.add_argument("--alg_queue_max_size", type=int, default=1)
        parser.add_argument("--buffer_name", type=str, default="replay_buffer")
        parser.add_argument("--buffer_warm_size", type=int, default=1000)
        parser.add_argument("--buffer_max_size", type=int, default=100000)
        parser.add_argument("--replay_batch_size", type=int, default=1024)
    else:
        raise ValueError
    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="mc_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=256)
    # Add noise to actions for better exploration
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={
            "mean": np.array([0], dtype=np.float32),
            "std": np.array([0.1], dtype=np.float32),
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
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=500)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=100)

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

    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    # Plot and save training figures
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])

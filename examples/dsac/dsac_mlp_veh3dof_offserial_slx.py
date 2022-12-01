#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: PPO algorithm, MLP, continuous version of cart pole, on serial trainer
#  Update Date: 2021-06-11, Li Jie: add PPO algorithm


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
    parser.add_argument("--env_id", type=str, default="simu_veh3dofconti", help="")
    parser.add_argument("--algorithm", type=str, default="DSAC", help="")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", default=2099945076, help="Disable CUDA")

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--action_type", type=str, default="continu")
    parser.add_argument("--is_render", type=bool, default=False)
    parser.add_argument(
        "--is_adversary", type=bool, default=False, help="Adversary training"
    )
    parser.add_argument(
        "--is_constrained", type=bool, default=False, help="Adversary training"
    )

    parser.add_argument("--ref_A", type=list, default=[0.3, 0.8, 1.5])  # dim(State)
    parser.add_argument("--ref_T", type=list, default=[100., 200., 400.])  # dim(State)
    parser.add_argument("--ref_fai", type=list, default=[0, np.pi / 6, np.pi / 3])  # dim(State)
    parser.add_argument("--ref_V", type=float, default=20.)  # dim(Action)
    parser.add_argument("--ref_info", type=str, default="None")  # dim(State)
    parser.add_argument("--ref_horizon", type=int, default=20)  # dim(Action)
    parser.add_argument("--Max_step", type=int, default=2000)  # dim(Action)
    parser.add_argument("--act_repeat", type=int, default=5)
    parser.add_argument("--obs_scaling", type=list, default=[0.001, 1, 1, 1, 2.4, 2])
    parser.add_argument("--act_scaling", type=list, default=[10, 1 / 1000, 1 / 1000])
    parser.add_argument("--act_max", type=list, default=[10 * np.pi / 180, 3000, 3000])
    parser.add_argument("--punish_done", type=float, default=0.)
    parser.add_argument("--rew_bias", type=float, default=2.5)
    parser.add_argument("--rew_bound", type=float, default=5)
    parser.add_argument("--punish_Q", type=list, default=[0.5, 0.5, 5, 0.25])
    parser.add_argument("--punish_R", type=list, default=[2.5, 5e-7, 5e-7])
    parser.add_argument("--rand_bias", type=list, default=[200, 1.5, 1.5, 0.1, np.pi / 18, 0.01]) ##[200, 2, 4, 0.1, np.pi / 18, 0.01]
    parser.add_argument("--rand_center", type=list, default=[0, 0, 20., 0, 0, 0])
    parser.add_argument("--done_range", type=list, default=[6., 6., np.pi / 6])

    ################################################
    # 2.1 Parameters of value approximate function
    # Options: StateValue/ActionValue/ActionValueDis
    parser.add_argument("--value_func_name", type=str, default="ActionValueDistri")
    parser.add_argument("--value_func_type", type=str, default="MLP")
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256])
    # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
    parser.add_argument("--value_hidden_activation", type=str, default="gelu")
    # Output Layer: linear
    parser.add_argument("--value_output_activation", type=str, default="linear")
    parser.add_argument("--value_min_log_std", type=int, default=-0.1)
    parser.add_argument("--value_max_log_std", type=int, default=4)

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
        parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
        # Hidden Layer Options: relu/gelu/elu/sigmoid/tanh
        parser.add_argument("--policy_hidden_activation", type=str, default="gelu")
        # Output Layer: tanh
        parser.add_argument("--policy_output_activation", type=str, default="linear")
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=1)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=3e-4)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)
    parser.add_argument("--alpha_learning_rate", type=float, default=1e-4)
    ## Special parameters
    parser.add_argument("--delay_update", type=int, default=2, help="")
    parser.add_argument("--TD_bound", type=float, default=10)
    parser.add_argument("--bound", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--reward_scale", type=float, default=1)
    parser.add_argument("--auto_alpha", type=bool, default=True)

    ################################################
    # 4. Parameters for trainer
    # Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer")
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=50000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--ini_network_dir", type=str, default=None)
    # 4.3. Parameters for off_serial_trainer
    if trainer_type == "off_serial_trainer":
        parser.add_argument("--buffer_name", type=str, default="replay_buffer")
        # Size of collected samples before training
        parser.add_argument("--buffer_warm_size", type=int, default=1000)
        # Max size of reply buffer
        parser.add_argument("--buffer_max_size", type=int, default=int(1e5))
        # Batch size of replay samples from buffer
        parser.add_argument("--replay_batch_size", type=int, default=256)
        # Period of sync central policy of each sampler
        parser.add_argument("--sampler_sync_interval", type=int, default=1)
    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=8)
    # Add noise to actions for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 7. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)

    ################################################
    # 8. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=500)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=200)

    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # alg.set_parameters({"reward_scale": 0.1, "gamma": 0.99, "tau": 0.05})
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
    # plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])


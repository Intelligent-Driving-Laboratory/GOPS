#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for sac + veh3dofconti + mlp + off_serial
#  Update Date: 2021-06-11, Yang Yujie: create example


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
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv

os.environ["OMP_NUM_THREADS"] = "4"

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="simu_veh3dofconti", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="SAC", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", default=2099945076, help="Disable CUDA")

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")

    parser.add_argument("--ref_A", type=list, default=[0.3, 0.8, 1.5], help="dim(State)")
    parser.add_argument("--ref_T", type=list, default=[100.0, 200.0, 400.0], help="dim(State)")
    parser.add_argument("--ref_fai", type=list, default=[0, np.pi / 6, np.pi / 3], help="dim(State)")
    parser.add_argument("--ref_V", type=float, default=20.0, help="dim(Action)")
    parser.add_argument("--ref_info", type=str, default="Pos", help="dim(State)")
    parser.add_argument("--ref_horizon", type=int, default=20, help="dim(Action)")
    parser.add_argument("--Max_step", type=int, default=2000, help="dim(Action)")
    parser.add_argument("--act_repeat", type=int, default=5)
    parser.add_argument("--obs_scaling", type=list, default=[0.001, 1, 1, 1, 2.4, 2])
    parser.add_argument("--act_scaling", type=list, default=[10, 1 / 1000, 1 / 1000])
    parser.add_argument("--act_max", type=list, default=[10 * np.pi / 180, 3000, 3000])
    parser.add_argument("--punish_done", type=float, default=0.0)
    parser.add_argument("--rew_bias", type=float, default=2.5)
    parser.add_argument("--rew_bound", type=float, default=5)
    parser.add_argument("--punish_Q", type=list, default=[0.5, 0.5, 5, 0.25])
    parser.add_argument("--punish_R", type=list, default=[2.5, 5e-7, 5e-7])
    parser.add_argument(
        "--rand_bias", type=list, default=[200, 1.5, 1.5, 0.1, np.pi / 18, 0.01]
    )  ##[200, 2, 4, 0.1, np.pi / 18, 0.01]
    parser.add_argument("--rand_center", type=list, default=[0, 0, 20.0, 0, 0, 0])
    parser.add_argument("--done_range", type=list, default=[6.0, 6.0, np.pi / 6])

    ################################################
    # 2.1 Parameters of value approximate function
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValue",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
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
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=1)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=1e-3)
    parser.add_argument("--q_learning_rate", type=float, default=1e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=1e-3)
    parser.add_argument("--alpha_learning_rate", type=float, default=1e-3)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=16000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--ini_network_dir", type=str, default=None)

    # 4.1. Parameters for off_serial_trainer
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=int(1e3))
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=int(1e5))
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)
    # Period of sync central policy of each sampler
    parser.add_argument("--sampler_sync_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=8)
    # Add noise to actions for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=5000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=200)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    alg.set_parameters({"tau": 0.05})
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

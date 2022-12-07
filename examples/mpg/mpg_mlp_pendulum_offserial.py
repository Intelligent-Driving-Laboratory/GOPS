#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for mpg + pendulum + mlp + off_serial
#  Update Date: 2022-06-05, Guan Yang: create example

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
    parser.add_argument("--env_id", type=str, default="gym_pendulum", help="id of environment")
    parser.add_argument("--algorithm", type=str, default="MPG")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    ################################################
    # 1. Parameters for environment
    parser.add_argument("--obsv_dim", type=int, default=None, help="dim(State)")
    parser.add_argument("--action_dim", type=int, default=None, help="dim(Action)")
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
    parser.add_argument("--value_func_name", type=str, default="ActionValue",
                        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri")
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument("--value_hidden_activation", type=str, default="relu",
                        help="Options: relu/gelu/elu/selu/sigmoid/tanh")
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")

    # 2.2 Parameters of policy approximate function
    parser.add_argument("--policy_func_name", type=str, default="DetermPolicy",
                        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy")
    parser.add_argument("--policy_func_type", type=str, default="MLP",
                        help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    parser.add_argument("--policy_act_distribution", type=str, default="default",
                        help="Options: default/TanhGaussDistribution/GaussDistribution")
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument(
        "--policy_output_activation", type=str, default="linear", help="Options: linear/tanh"
    )

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=1e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=5e-4)
    # special parameter
    parser.add_argument("--pge_method", type=str, default='mixed_weight', help='mixed_weight/mixed_state')
    pge_method = parser.parse_known_args()[0].pge_method
    if pge_method == 'mixed_weight':
        parser.add_argument("--eta", type=float, default=0.3)
        parser.add_argument("--terminal_iter", type=float, default=1e8)  # always use model because model is accurate
    else:
        assert pge_method == 'mixed_state'
        parser.add_argument("--kappa", type=float, default=0.5)
    parser.add_argument("--forward_step", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--delay_update", type=int, default=1, help="")
    # Reward = reward_scale * environment.Reward
    parser.add_argument("--reward_scale", type=float, default=0.1)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument("--trainer", type=str, default="off_serial_trainer",
                        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer")
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=5000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--ini_network_dir", type=str, default=None,
                        help="path of saved approximate functions, if specified, the saved approximate functions "
                             "will be loaded before training")

    # 4.1. Parameters for off_serial_trainer
    parser.add_argument("--buffer_name", type=str, default="replay_buffer",
                        help="Options:replay_buffer/prioritized_replay_buffer")
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=1000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=100000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)
    # Period of sampling
    parser.add_argument("--sampler_sync_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler",
                        help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=8)
    # Add noise to action for better exploration
    parser.add_argument(
        "--noise_params",
        type=dict,
        default={
            "mean": np.array([0], dtype=np.float32),
            "std": np.array([0.2], dtype=np.float32),
        },
    )

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=5000)
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

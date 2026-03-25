#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for fpisac + veh2dof_tracking_error + mlp + off_serial
#  Update Date: 2026-3-25, Yujie Yang: create example

import argparse

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument("--env_id", type=str, default="veh2dof_tracking_error")
    parser.add_argument("--algorithm", type=str, default="FPISAC")
    parser.add_argument("--pre_horizon", type=int, default=30)
    parser.add_argument("--enable_cuda", default=False)
    parser.add_argument("--seed", default=None, help="seed")

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--is_render", type=bool, default=False)
    parser.add_argument("--is_adversary", type=bool, default=False)

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValue",
        help="Options: StateValue/ActionValue/ActionValueDis/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP")
    parser.add_argument("--value_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument("--value_hidden_activation", type=str, default="relu")

    # 2.2 Parameters of feasibility approximate function
    parser.add_argument("--feasibility_func_name", type=str, default="ActionValue")
    parser.add_argument("--feasibility_func_type", type=str, default="MLP")
    parser.add_argument("--feasibility_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument("--feasibility_hidden_activation", type=str, default="relu")

    # 2.3 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="StochaPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument("--policy_func_type", type=str, default="MLP")
    parser.add_argument("--policy_act_distribution", type=str, default="TanhGaussDistribution")
    parser.add_argument("--policy_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument("--policy_hidden_activation", type=str, default="relu")
    parser.add_argument("--policy_min_log_std", type=float, default=-20.)
    parser.add_argument("--policy_max_log_std", type=float, default=2.)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=3e-4)
    parser.add_argument("--feasibility_learning_rate", type=float, default=3e-4)
    parser.add_argument("--policy_learning_rate", type=float, default=3e-4)
    parser.add_argument("--alpha_learning_rate", type=float, default=3e-4)
    parser.add_argument("--penalty", type=float, default=10.)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    parser.add_argument("--max_iteration", type=int, default=100000)
    parser.add_argument("--ini_network_dir", type=str, default=None)

    # 4.1. Parameters for off_serial_trainer
    parser.add_argument("--buffer_name", type=str, default="replay_buffer")
    parser.add_argument("--buffer_warm_size", type=int, default=1000)
    parser.add_argument("--buffer_max_size", type=int, default=100000)
    parser.add_argument("--replay_batch_size", type=int, default=256)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler")
    parser.add_argument("--sample_batch_size", type=int, default=1)
    parser.add_argument("--sample_interval", type=int, default=1)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--apprfunc_save_interval", type=int, default=20000)
    parser.add_argument("--log_save_interval", type=int, default=1000)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)
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

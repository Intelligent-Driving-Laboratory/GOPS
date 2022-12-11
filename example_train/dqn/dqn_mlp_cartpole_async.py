#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for dqn + cartpole + mlp + async trainer
#  Update Date: 2021-01-03, Yuxuan Jiang & Guojian Zhan: create example


import argparse
import multiprocessing

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
    parser.add_argument("--env_id", type=str, default="gym_cartpole")
    parser.add_argument("--algorithm", type=str, default="DQN")
    parser.add_argument("--enable_cuda", default=False)

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--action_type", type=str, default="discret")
    parser.add_argument("--is_render", type=bool, default=False)

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="ActionValueDis"
    )
    parser.add_argument("--value_func_type", type=str, default="MLP")
    value_func_type = parser.parse_known_args()[0].value_func_type

    parser.add_argument("--value_hidden_sizes", type=list, default=[64, 64])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="relu"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear")

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="DetermPolicyDis"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="default"
    )

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=1e-3)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_async_trainer"
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=20000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    # 4.1. Parameters for off_async_trainer
    import ray

    ray.init()
    parser.add_argument("--num_algs", type=int, default=1)
    parser.add_argument("--num_samplers", type=int, default=2)
    parser.add_argument("--num_buffers", type=int, default=1)
    cpu_core_num = multiprocessing.cpu_count()
    num_core_input = (
        parser.parse_known_args()[0].num_algs
        + parser.parse_known_args()[0].num_samplers
        + parser.parse_known_args()[0].num_buffers
        + 2
    )
    if num_core_input > cpu_core_num:
        raise ValueError("The number of core is {}, but you want {}!".format(cpu_core_num, num_core_input))
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=1000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=100000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=64)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=4)
    # Add noise to actions for better exploration
    parser.add_argument("--noise_params", type=dict, default={"epsilon": 0.25})

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
    parser.add_argument("--apprfunc_save_interval", type=int, default=500)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=100)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    for alg_id in alg:
        alg_id.set_parameters.remote({"gamma": 0.99, "tau": 0.2})
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
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

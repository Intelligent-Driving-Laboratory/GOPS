#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: example for fhadp2 + idsim + mlp + off_serial
#  Update Date: 2022-9-21, Jiaxin Gao: create example


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
    parser.add_argument("--env_id", type=str, default="pyth_idsim", help="id of environment")
    MAP_ROOT = 'YOUR_MAP_ROOT'
    pre_horizon = 30
    env_config_param = {
        "use_render": False,
        "seed": 1,
        "actuator": "ExternalActuator",
        "scenario_reuse": 10,
        "num_scenarios": 20, 
        "detect_range": 60,
        "choose_vehicle_retries": 10,
        "scenario_root": MAP_ROOT,
        "scenario_selector": '1',
        "extra_sumo_args": ("--start", "--delay", "200"),
        "warmup_time": 5.0,
        "ignore_traffic_lights": False,
        "incremental_action": True,
        "action_lower_bound": (-0.5, -0.03),
        "action_upper_bound": ( 0.2, 0.03),
        "real_action_lower_bound": (
            -3.0,
            -0.45
        ),
        "real_action_upper_bound": (
            0.8,
            0.45
        ),
        "obs_num_surrounding_vehicles": {
            "passenger": 5,
            "bicycle": 0,
            "pedestrian": 0,
        },
        "ref_v": 8.0,
        "ref_length": 48.0,
        "obs_num_ref_points": 2 * pre_horizon + 1,
        "obs_ref_interval": 0.8,
        "vehicle_spec": (1880.0, 1536.7, 1.13, 1.52, -128915.5, -85943.6, 20.0, 0.0),
        "singleton_mode": "reuse",
        "seed": 1
    }
    model_config = {
        "N": pre_horizon,
        "full_horizon_sur_obs": False,
        "ahead_lane_length_min": 6.0,
        "ahead_lane_length_max": 30.0,
        "v_discount_in_junction_straight": 0.75,
        "v_discount_in_junction_left_turn": 0.5,
        "v_discount_in_junction_right_turn": 0.375,
        "num_ref_lines": 3,
        "dec_before_junction": 0.8,
        "ego_length": 5.0,
        "ego_width": 1.8,
        "safe_dist_incremental": 1.5,

        "num_ref_points": pre_horizon + 1,
        "ego_feat_dim": 7, # vx, vy, r, last_last_acc, last_last_steer, last_acc, last_steer
        "per_sur_state_dim": 6, # x, y, phi, speed, length, width
        "per_sur_state_withinfo_dim": 7, # x, y, phi, speed, length, width, mask
        "per_sur_feat_dim": 5, # x, y, cos(phi), sin(phi), speed
        "per_ref_feat_dim": 5, # x, y, cos(phi), sin(phi), speed
        "real_action_upper": (
            (
                0.8,
                0.45
            )
        ),
        "real_action_lower": (
            (
                -3.0,
                -0.45
            )
        ),
        "Q": (
            0.0,
            16.0,
            750.0,
            2.0,
            2.0,
            60.0
        ),
        "R0": (
            0,
            40
        ),
        "R1": (
            5.0,
            5.0
        ),
        "R2": (
            5.0,
            5.0
        ),
        "P": 2000.0,
        "ref_v_lane": 8.0,
        "filter_num": 5
    }
    parser.add_argument("--env_config", type=dict, default=env_config_param)
    parser.add_argument("--env_model_config", type=dict, default=model_config)

    parser.add_argument("--algorithm", type=str, default="FHADP2", help="RL algorithm")
    parser.add_argument("--enable_cuda", default=False, help="Enable CUDA")
    parser.add_argument("--seed", default=2099945076, help="seed")
    parser.add_argument("--pre_horizon", type=int, default=pre_horizon)

    # 1. Parameters for environment
    parser.add_argument("--action_type", type=str, default="continu", help="Options: continu/discret")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")
    parser.add_argument("--is_constrained", type=bool, default=False, help="Adversary training")
    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="StateValue",
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
        default="FiniteHorizonFullPolicy",
        help="Options: None/DetermPolicy/FiniteHorizonPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="default",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256, 256])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--policy_output_activation", type=str, default="linear", help="Options: linear/tanh")

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=1e-3)
    parser.add_argument("--policy_learning_rate", type=float, default=1e-3)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer, off_sync_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=200000)
    trainer_type = parser.parse_known_args()[0].trainer
    parser.add_argument("--ini_network_dir", type=str, default=None)

    # import ray
    # ray.init('local')
    # parser.add_argument("--num_algs", type=int, default=4, help="number of algs")
    # parser.add_argument("--num_samplers", type=int, default=1, help="number of samplers")
    # parser.add_argument("--num_buffers", type=int, default=1, help="number of buffers")

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
    parser.add_argument("--sample_batch_size", type=int, default=128)
    # Add noise to actions for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator")
    parser.add_argument("--num_eval_episode", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=10000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=1000)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    args["env"] = env
    alg = create_alg(**args)  # create appr_model in algo **vars(args)
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

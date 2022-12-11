#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: template for running policy by PolicyRunner
#  Update: 2022-12-10, Zhilong Zheng: create example template

import argparse

from gops.sys_simulator.sys_run import PolicyRunner

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Parameters for policies to be run
    parser.add_argument("--log_policy_dir_list", type=list, default=["../results/INFADP/lqs4a2"], help="directory of trained policy")
    parser.add_argument("--trained_policy_iteration_list", type=list, default=["115000_opt"], help="iteration of trained policy")

    ################################################
    # Parameters for results saving and figures drawing
    parser.add_argument("--save_render", type=bool, default=False, help="save environment animation or not")
    parser.add_argument("--plot_range", type=list, default=[0, 100], help="customize plot range")
    parser.add_argument("--legend_list", type=list, default=["INFADP-115000"], help="legends of figures")
    parser.add_argument("--constrained_env", type=bool, default=False, help="constrainted environment or not")
    parser.add_argument("--is_tracking", type=bool, default=False, help="tracking problem or not")
    parser.add_argument("--use_dist", type=bool, default=False, help="use adversarial action or not")
    parser.add_argument("--dt", type=float, required=False, help="time interval between steps")

    ################################################
    # Parameters for environment
    parser.add_argument("--is_init_info", type=bool, default=True, help="customize initial information or not")
    parser.add_argument("--init_info", type=dict, default={"init_state": [0.5, 0.2, 0.5, 0.1]}, help="initial information")
    
    ################################################
    # Parameters for optimal controller
    parser.add_argument("--use_opt", type=bool, default=True, help="use optimal solution for comparison or not")
    parser.add_argument("--opt_args", type=dict, default={
        "opt_controller_type": "MPC",
        "num_pred_step": 50,
        "gamma": 0.99,
        "minimize_options": {"max_iter": 200, "tol": 1e-4,
                             "acceptable_tol": 1e-2,
                             "acceptable_iter": 10, },
    }, help="arguments of optimal solution solver")

    ################################################
    # Parameters for obs and action noise
    parser.add_argument("--obs_noise_type", type=str, required=False, choices=["normal", "uniform"], help="type of observation noise")
    parser.add_argument("--obs_noise_data", type=list, required=False, help="Mean and Standard deviation of Normal distribution or Upper and Lower bounds of Uniform distribution")
    parser.add_argument("--action_noise_type", type=str, required=False, choices=["normal", "uniform"], help="type of action noise")
    parser.add_argument("--action_noise_data", type=list, required=False, help="Mean and Standard deviation of Normal distribution or Upper and Lower bounds of Uniform distribution")
    
    ################################################
    # call PolicyRunner
    args = vars(parser.parse_args())
    runner = PolicyRunner(**args)
    runner.run()
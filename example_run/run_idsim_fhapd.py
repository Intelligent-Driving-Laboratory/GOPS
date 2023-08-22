 #  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system
#  Update: 2022-12-05, Congsheng Zhang: create file


from gops.sys_simulator.sys_run import PolicyRunner
import numpy as np

runner = PolicyRunner(
    log_policy_dir_list=["../results/pyth_idsim/FHADP2_230816-142507",
                         "../results/pyth_idsim/FHADP_230816-202627"],
    # trained_policy_iteration_list=["54000_opt"],
    trained_policy_iteration_list=["54000_opt","11100_opt"],
    is_init_info=True,
    init_info={}, # ref_num = [0, 1, 2,..., 7]
    save_render=False,
    # legend_list=["FHADP2"],
    legend_list=["FHADP2","FHADP"],
    use_opt=True, # Use optimal solution for comparison
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 30,
        "gamma": 0.99,
        "mode": "shooting",
        "minimize_options": {
            "max_iter": 200,
            "tol": 1e-4,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 10,
        },
        "use_terminal_cost": False,
        "use_MPC_for_general_env": True,
    },
    constrained_env=False,
    is_tracking=False,
    dt=0.1,
)

runner.run()

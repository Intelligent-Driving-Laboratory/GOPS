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
    log_policy_dir_list=["./results/pyth_veh3dofconti_detour/DSAC_230910-000333"],
    trained_policy_iteration_list=["40000"],
    is_init_info=True,
    init_info={"init_state": [0.0, -0.2, 0.0, 0.0, 0, 0], "ref_time": 0.0,
               "ref_num": 9}, # ref_num = [0, 1, 2,..., 7]
    save_render=True,
    legend_list=["DSAC-best"],
    use_opt=True, # Use optimal solution for comparison
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 20,
        "gamma": 0.99,
        "mode": "shooting",
        "minimize_options": {
            "max_iter": 50,
            "tol": 1e-4,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 10,
        },
        "use_terminal_cost": False,
    },
    constrained_env=False,
    is_tracking=True,
    dt=0.1,
)

runner.run()

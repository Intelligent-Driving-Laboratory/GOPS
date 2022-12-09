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
    log_policy_dir_list=["../../results/SPIL/221203-160948"] * 1,
    trained_policy_iteration_list=["10000"],
    is_init_info=True,
    init_info={"init_state": [1, -0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 2.5, -1, np.pi / 2 + 0.1, 0.25, 0]},
    save_render=False,
    legend_list=["9000"],
    use_opt=False,
    constrained_env=True,
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 50,
        "gamma": 1,
        "minimize_options": {
            "max_iter": 200,
            "tol": 1e-3,
            "acceptable_tol": 1e0,
            "acceptable_iter": 10,
            # "print_level": 5,
        },
        "use_terminal_cost": False,
        # "terminal_cost": terminal_cost,
    },
)

runner.run()

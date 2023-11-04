#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system


from gops.sys_simulator.sys_run import PolicyRunner


runner = PolicyRunner(
    log_policy_dir_list=[
        # "../results/veh3dof_tracking/<ALGORITHM>_<DATETIME>"
        "PATH_TO_YOUR_RESULT_DIR",
    ],
    trained_policy_iteration_list=[
        # e.g., "1000", "1000_opt"
        "ITERATION_NUM",
    ],
    is_init_info=True,
    init_info={
        # parameters of env.reset()
        "init_state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "ref_time": 0.0,
        "ref_num": 0,
    },
    save_render=True,
    legend_list=[
        # e.g., "FHADP"
        "ALGORITHM_NAME",
    ],
    use_opt=True,
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 30,
        "gamma": 1.0,
        "mode": "collocation",
        "minimize_options": {
            "max_iter": 100,
            "tol": 1e-4,
        },
        "use_terminal_cost": False,
        "use_MPC_for_general_env": True,
    },
    constrained_env=True,
    is_tracking=True,
    dt=0.1,
)

runner.run()

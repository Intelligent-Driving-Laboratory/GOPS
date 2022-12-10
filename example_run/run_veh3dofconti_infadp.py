#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system, tracking environment, compared with MPC controller
#  Update: 2022-12-05, Congsheng Zhang: create file


from gops.sys_simulator.sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["../results/INFADP/veh3dofconti"]*2,
    trained_policy_iteration_list=["4000", "1300_opt"],
    is_init_info=True,
    init_info={"init_state": [0.0, 0, 0.0, 5.0, 0, 0], "ref_time": 0.0, "path_num": 1, "u_num": 0},
    save_render=False,
    legend_list=["INFADP-4000", "INFADP-1300"],
    use_opt=True,
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 10,
        "gamma": 0.99,
        "mode": "shooting",
        "minimize_options": {
            "max_iter": 200,
            "tol": 1e-4,
            "acceptable_tol": 1e-2,
            "acceptable_iter": 10,
            # "print_level": 5,
        },
        "use_terminal_cost": False,
    },
    constrained_env=False,
    is_tracking=True,
    dt=0.1,
)

runner.run()

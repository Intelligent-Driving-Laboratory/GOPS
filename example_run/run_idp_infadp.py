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


runner = PolicyRunner(
    log_policy_dir_list=["../results/INFADP/idpendulum"],
    trained_policy_iteration_list=["90000_opt"],
    is_init_info=True,
    init_info={"init_state": [0.0, 0.05, -0.05, 0.05, 0.1, -0.1]},
    save_render=False,
    legend_list=["INFADP"],
    use_opt=True,  # Use optimal solution for comparison
    opt_args={
        "opt_controller_type": "MPC",  # OPT or MPC
        "num_pred_step": 1,
        "gamma": 0.99,
        "minimize_options": {"max_iter": 200, "tol": 1e-3,
                             "acceptable_tol": 1e0,
                             "acceptable_iter": 10,},
        "use_terminal_cost": False,
    },
    dt=0.01, # time interval between steps
)

runner.run()

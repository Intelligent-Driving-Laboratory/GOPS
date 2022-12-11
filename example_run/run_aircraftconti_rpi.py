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
    log_policy_dir_list=["../results/RPI/aircraftconti"] * 2,  # directory of trained policy
    trained_policy_iteration_list=["40", "50"],  # iteration number of trained policy
    is_init_info=True,  # customize initial information
    init_info={"init_state": [0.3, -0.5, 0.2]},  # initial state
    save_render=False,  # save environment animation
    legend_list=["RPI-40", "RPI-50"],  # legends of figures
    use_opt=True,  # use optimal solution for comparison
    constrained_env=False,  # constraint environment
    is_tracking=False,  # tracking problem
    opt_args={"opt_controller_type": "OPT"},  # arguments of optimal solution
    dt=None,  # time interval between steps
)

runner.run()

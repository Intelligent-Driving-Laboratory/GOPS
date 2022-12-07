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
    log_policy_dir_list=["../../results/SAC/idp_221017-174348"] * 2,
    trained_policy_iteration_list=["24000", "27000"],
    is_init_info=False,
    init_info={"init_state": np.array([0.063, 0.0076, -0.0029, 0.046, 0.1, -0.2], dtype=np.float64)},
    save_render=False,
    legend_list=["24000", "27000"],
    dt=0.05,
    plot_range=[0, 50],
    use_opt=True,
)

runner.run()

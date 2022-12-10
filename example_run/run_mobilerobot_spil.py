#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system, constrained environment
#  Update: 2022-12-05, Congsheng Zhang: create file


from gops.sys_simulator.sys_run import PolicyRunner
import numpy as np

runner = PolicyRunner(
    log_policy_dir_list=["../results/SPIL/mobilerobot"],
    trained_policy_iteration_list=["16500_opt"],
    is_init_info=True,
    init_info={"init_state": [1, -0.5, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 5.5, -2.5, np.pi / 2, 0.2, 0]},
    save_render=False,
    legend_list=["SPIL"],
    use_opt=False,
    constrained_env=True,
)

runner.run()

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: run a closed-loop system
#  Update: 2022-12-09, Shengbo Li: create file


from gops.utils.common_utils import get_args_from_json
from gops.sys_simulator.sys_run import PolicyRunner
import torch
from gops.algorithm.infadp import ApproxContainer
import os
import argparse


#####################################
# Run Systems for Comparison
runner = PolicyRunner(
    log_policy_dir_list=["../results/FHADP/lqs3a1",
                         "../results/FHADP/lqs3a1"],
    trained_policy_iteration_list=["5400_opt",
                                   "1000"],
    is_init_info=True,
    init_info={"init_state": [0.5, 0.2, 0.5]},
    save_render=False,
    legend_list=["FHADP-5400", "FHADP-1000"],
    use_opt=True,
    opt_args={"opt_controller_type": "OPT"},
    dt=None,  # time interval between steps
)

runner.run()

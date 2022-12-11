#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Template for checking open-loop dynamics whether its time-discretizaiton is reasonable!
#  Update: 2020-12-10, Xujie Song: create example template

import argparse

from gops.env.inspector.env_dynamic_checker import check_dynamic

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Parameters for environment to tested
    parser.add_argument("--env_info", type=dict, default={"env_id": "pyth_lq", "lq_config": "s4a2"}, help="env info")
    parser.add_argument("--traj_num", type=int, default=5, help="number of trajectories sampled")
    parser.add_argument("--init_info", type=dict, default=None, help="initialization info of env")

    ################################################
    # Call open-loop checker
    args = vars(parser.parse_args())
    check_dynamic(**args)

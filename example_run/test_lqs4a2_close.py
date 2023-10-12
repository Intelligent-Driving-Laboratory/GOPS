#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: check close-loop dynamic of linear dynamics with 4 states and 2 actions.
#  Update: 2022-12-05, Xujie Song: create file

from gops.env.inspector.env_dynamic_checker import check_dynamic

check_dynamic(
    env_info={"env_id": "pyth_lq", "lq_config": "s4a2"},
    traj_num=3,
    log_policy_dir="./results/INFADP/lqs4a2_mlp",
    policy_iteration="6000",
)

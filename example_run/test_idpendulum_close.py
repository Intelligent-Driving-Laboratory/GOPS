#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: check close-loop dynamic of pyth_idpendulum.
#  Update: 2022-12-05, Xujie Song: create file

from gops.env.inspector.env_dynamic_checker import check_dynamic

check_dynamic(
    env_info={"env_id": "pyth_idpendulum"},
    traj_num=5,
    log_policy_dir="../results/FHADP/idpendulum",
    policy_iteration="54000_opt",
)

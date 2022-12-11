#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: check the closed-loop dynamic of gym_pendulum, draw the figures of first/second-order difference of state.
#               figures will be saved in 'figures' folder.
#  Update: 2022-12-05, Xujie Song: create file

from gops.env.inspector.env_dynamic_checker import check_dynamic


check_dynamic(
    env_info={"env_id": "gym_pendulum"},
    traj_num=2,
    log_policy_dir="../results/DDPG/gym_pendulum",
    policy_iteration="8000",
)

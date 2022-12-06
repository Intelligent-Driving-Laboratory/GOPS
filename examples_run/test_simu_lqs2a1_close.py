#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: check the close-loop dynamic of simu_lq_s2a1, draw the figures of first/second-order difference of state.
#               figures will be saved in 'figures' folder.
#  Update: 2022-12-05, Xujie Song: create file

from gops.env.tools.env_dynamic_checker import check_dynamic, load_args


check_dynamic(env_info=load_args('./results/DDPG/simu_lqs2a1'), 
              traj_num=3,
              init_info={"init_state": [[0.5, -0.3], [1, -1], [-1, 0.7]]},
              log_policy_dir='./results/DDPG/simu_lqs2a1',
              policy_iteration='250000')
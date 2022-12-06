#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: check the close-loop dynamic of pyth_lq_s2a1, draw the figures of first/second-order difference of state.
#               figures will be saved in 'figures' folder.
#  Update: 2022-12-05, Xujie Song: create file

from gops.env.tools.env_dynamic_checker import check_dynamic


check_dynamic(env_info={'env_id':'pyth_lq', 'lq_config':'s2a1'},
              log_policy_dir='./results/INFADP/s2a1',
              policy_iteration='350000')
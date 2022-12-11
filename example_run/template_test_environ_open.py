#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Template for checking open-loop dynamics whether its time-discretizaiton is reasonable!
#  Update: 2022-12-10, Xujie Song: create example template

from gops.env.inspector.env_dynamic_checker import check_dynamic

check_dynamic(
    # dict 'env_info': env name and its config
    env_info={"env_id": "pyth_lq", "lq_config": "s4a2"},

    # int 'traj_num': number of trajectories sampled
    traj_num=3,

    # dict 'init_info': initialization info of env
    # Option 1: no init_info, reset by default
        init_info = None,
    # Option 2: reset by given state
    #   init_info = {"init_state": [[0.0, 0.0, 0.0, 0.0]]},
    # Option 3: reset by given state and ref trajectory (e.g. for veh3dofconti env)
    #   init_info={"init_state": [[0.0, 0.0, 0.0, 1, 0.1, 0.1]], "ref_num": [3]},
    # Option 4: up to init parameters for each env
)

from sys_run import PolicyRuner
import numpy as np

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/veh3dofconti_221017-210557"]*2,
    trained_policy_iteration_list=['3000','4000'],
    is_init_info=True,
    init_info={"init_state": [0., 0., 0., 10.,0,0], "ref_time": 0., "ref_num": 0},
    save_render=False,
    legend_list=[ 'INFADP-3000','INFADP-4000'],
    use_opt=True,
    constrained_env=False,
    is_tracking=True,
    # obs_noise_data=[[-0.1, -0.02, -0.1,-0.02,0,0]+[0]*20, [0.1, 0.02, 0.1,0.02,0,0]+[0]*20],
    # obs_noise_type='uniform',
    # action_noise_data=[[0], [0.0]],
    # action_noise_type='normal',
    dt=None)

runer.run()

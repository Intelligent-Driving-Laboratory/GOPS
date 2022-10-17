from sys_run import PolicyRuner
import numpy as np

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/veh2dofconti_221017-211644"]*2,
    trained_policy_iteration_list=['2000','3500'],
    is_init_info=True,
    init_info={"init_state": [1., 0.0, 0., 0.], "ref_time": 0., "ref_num": 0},
    save_render=False,
    legend_list=[ 'INFADP-2000','INFADP-3000'],
    use_opt=True,
    constrained_env=False,
    is_tracking=True,
    dt=None)

runer.run()

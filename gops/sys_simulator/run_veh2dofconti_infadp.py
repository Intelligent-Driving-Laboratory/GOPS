from sys_run import PolicyRuner
import numpy as np

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/221007-211601"],
    trained_policy_iteration_list=['4000'],
    is_init_info=True,
    init_info={"init_state": [0., 1., 0., 10., 0., 0.], "ref_init_time": 0., "ref_num": 1},
    save_render=False,
    legend_list=[ 'INFADP-500'],
    use_opt=False,
    constrained_env=False,
    is_tracking=True,
    dt=None)

runer.run()

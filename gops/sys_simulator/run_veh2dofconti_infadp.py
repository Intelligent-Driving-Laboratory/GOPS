from sys_run import PolicyRuner
import numpy as np

runer = PolicyRuner(
    log_policy_dir_list=["../../results/FHADP/220930-212359"],
    trained_policy_iteration_list=['500'],
    is_init_info=True,
    init_info={"init_state":[1.,0.,0.,0.], "t":0.,"ref_num":1},
    save_render=False,
    legend_list=[ 'INFADP-500'],
    use_opt=False,
    constrained_env=False,
    is_tracking=True,
    dt=None)

runer.run()

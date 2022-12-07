from sys_run import PolicyRunner
import numpy as np

runner = PolicyRunner(
    log_policy_dir_list=["../../results/INFADP/veh3dofconti_221116_dt_0_1"],
    trained_policy_iteration_list=["4000"],
    is_init_info=True,
    init_info={"init_state": [0.0, 0, 0.0, 3.0, 0, 0], "ref_time": 0.0, "path_num": 3, "u_num": 1},
    save_render=False,
    legend_list=["INFADP-4000"],
    use_opt=False,
    constrained_env=False,
    is_tracking=True,
    dt=0.1,
)

runner.run()

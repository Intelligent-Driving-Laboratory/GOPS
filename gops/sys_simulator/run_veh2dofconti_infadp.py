from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/220928-124545", "../../results/INFADP/220928-124545"],
    trained_policy_iteration_list=['3000', '3500'],
    is_init_state=False,
    init_state=[],
    save_render=False,
    legend_list=['INFADP-3000', 'INFADP-3500'],
    use_opt=False,
    constrained_env=False,
    is_tracking=True,
    dt=None)

runer.run()

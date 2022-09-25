from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/220925-094637", "../../results/INFADP/220925-094637"],
    trained_policy_iteration_list=['1500', '1900'],
    is_init_state=False,
    init_state=[],
    save_render=False,
    legend_list=['1500', '1900'],
    use_opt=False,
    constrained_env=False,
    is_tracking=True)

runer.run()

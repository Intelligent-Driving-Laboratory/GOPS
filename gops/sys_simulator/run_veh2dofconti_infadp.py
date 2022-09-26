from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/220926-154220"],
    trained_policy_iteration_list=['6000'],
    is_init_state=False,
    init_state=[],
    save_render=False,
    legend_list=['6000'],
    use_opt=False,
    constrained_env=False,
    is_tracking=True)

runer.run()

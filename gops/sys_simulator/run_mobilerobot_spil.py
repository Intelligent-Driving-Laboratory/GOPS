from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/SPIL/220923-235913"],
    trained_policy_iteration_list=['200'],
    is_init_info=False,
    init_info=[2, -0.1],
    save_render=False,
    # legend_list=['5000'],
    use_opt=False,
    constrained_env=True)

runer.run()

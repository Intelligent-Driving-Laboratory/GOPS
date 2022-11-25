from sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["../../results/SPIL/mobile_robot_221017-145318"]*1,
    trained_policy_iteration_list=['7900'],
    is_init_info=True,
    init_info={"init_state": [0.6, 0.3, 0.3, 0.2, 0.3, 0.0, 0.0, 0.0, 4.5, 0., -0.7, 0.36, 0.]},
    save_render=False,
    legend_list=['7900'],
    use_opt=False,
    plot_range= [0,50],
    constrained_env=True)

runner.run()

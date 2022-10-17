from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/SPIL/mobile_robot_221017-145318"]*2,
    trained_policy_iteration_list=['7000', '8000'],
    is_init_info=True,
    init_info={"init_state": [0.0, 0.0, 0, 0, 0, 0.0, 0., 0., 0., 0., 0., 0., 0.]},
    save_render=False,
    legend_list=['5000', '7000'],
    use_opt=True,
    constrained_env=True)

runer.run()

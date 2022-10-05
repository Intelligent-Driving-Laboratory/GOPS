from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/RPI/221005-233433"]*1,
    trained_policy_iteration_list=['40'],
    is_init_info=False,
    init_info={"init_state":[0.98, -0.5]},
    save_render=False,
    legend_list=['40'],
    use_opt=False)

runer.run()

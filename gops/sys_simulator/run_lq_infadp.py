from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/FHADP/221005-224236"]*1,
    trained_policy_iteration_list=['2000'],
    is_init_info=True,
    init_info={"init_state":[0.98, -0.5]},
    save_render=False,
    legend_list=['2000'],
    use_opt=True)

runer.run()

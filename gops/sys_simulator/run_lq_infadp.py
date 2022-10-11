from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/FHADP/221011-084859"]*1,
    trained_policy_iteration_list=['4000'],
    is_init_info=True,
    init_info={"init_state":[2, 1]},
    save_render=False,
    legend_list=['4000'],
    use_opt=True)

runer.run()

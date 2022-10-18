from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/FHADP/s6a3"]*2,
    trained_policy_iteration_list=['9000','9500'],
    is_init_info=True,
    init_info={"init_state":[0.1, 0.1,-0.1,0.2,-0.1,0.15]},
    save_render=False,
    legend_list=['9000','9500'],
    dt=0.05,
    use_opt=True)

runer.run()

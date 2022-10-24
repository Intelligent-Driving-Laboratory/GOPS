from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/s2a1"]*2,
    trained_policy_iteration_list=['9000','9500'],
    is_init_info=True,
    init_info={"init_state":[0.5, -0.5]},
    save_render=False,
    legend_list=['4000','5000'],
    dt=0.05,
    use_opt=True)

runer.run()

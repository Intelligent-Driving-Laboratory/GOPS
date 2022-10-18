from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/s3a1"]*2,
    trained_policy_iteration_list=['9000','10000'],
    is_init_info=True,
    init_info={"init_state":[0.5, 0.5,0.5]},
    save_render=False,
    legend_list=['9000','9500'],
    dt=0.1,
    use_opt=True)

runer.run()

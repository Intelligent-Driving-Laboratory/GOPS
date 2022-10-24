from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/s3a1"]*2,
    trained_policy_iteration_list=['19500','20000'],
    is_init_info=True,
    init_info={"init_state":[0.5, 0.1,0.1]},
    save_render=False,
    legend_list=['19500','20000'],
    dt=0.1,
    use_opt=True)

runer.run()

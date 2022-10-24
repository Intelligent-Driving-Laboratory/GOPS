from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/s2a1"]*2,
    trained_policy_iteration_list=['10000', '9000'],
    is_init_info=True,
    init_info={"init_state":[0.5, 0.1]},
    save_render=False,
    legend_list=['INFADP-10000', 'INFADP-9000'],
    use_opt=True)

runer.run()
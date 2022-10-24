from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/s6a3"]*2,
    trained_policy_iteration_list=['99000', '98000'],
    is_init_info=True,
    init_info={"init_state":[0.05, 0.1, 0, 0, 0, 0.1]},
    save_render=False,
    legend_list=['INFADP-99000', 'INFADP-98000'],
    use_opt=True)

runer.run()

from sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["../../results/INFADP/s5a1"]*2,
    trained_policy_iteration_list=['180000', '190000'],
    is_init_info=True,
    init_info={"init_state":[0.1, 0.2, 0, 0.1, 0]},
    save_render=False,
    legend_list=['INFADP-180000', 'INFADP-190000'],
    use_opt=True)

runner.run()

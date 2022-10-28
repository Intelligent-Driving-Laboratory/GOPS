from sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["../../results/INFADP/s4a2"]*2,
    trained_policy_iteration_list=['100000', '99000'],
    is_init_info=True,
    init_info={"init_state":[0.5, 0.2, 0.5, 0.2]},
    save_render=False,
    legend_list=['INFADP-100000', 'INFADP-99000'],
    use_opt=True)

runner.run()

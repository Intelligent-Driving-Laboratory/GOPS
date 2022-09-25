from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/220919-152944", "../../results/INFADP/220919-152944", "../../results/INFADP/220919-152944"],
    trained_policy_iteration_list=['5000', '6000', '3500'],
    is_init_state=False,
    init_state=[2, -0.1],
    save_render=False,
    legend_list=['5000', '6000', '3500'],
    use_opt=True)

runer.run()

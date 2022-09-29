from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/FHADP/mlp_s2a1_80step"]*1,
    trained_policy_iteration_list=['6000'],
    is_init_state=True,
    init_state=[2,-1],
    save_render=False,
    legend_list=['6000'],
    use_opt=True)

runer.run()

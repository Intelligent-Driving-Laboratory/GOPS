from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/220929-004343"]*1,
    trained_policy_iteration_list=['5000'],
    is_init_state=True,
    init_state=[0.01, -0.05,0.0,0.0,-0.0],
    save_render=False,
    legend_list=['5000'],
    use_opt=True)

runer.run()

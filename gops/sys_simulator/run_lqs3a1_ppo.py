from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/PPO/221018-200154"]*2,
    trained_policy_iteration_list=['100','220_opt'],
    is_init_info=True,
    init_info={"init_state":[0.5, 0.5,0.5]},
    save_render=False,
    legend_list=['300','223_opt'],
    dt=0.1,
    use_opt=True)

runer.run()

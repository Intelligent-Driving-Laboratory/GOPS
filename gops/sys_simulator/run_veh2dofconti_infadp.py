from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/220929-103609"]*2,
    trained_policy_iteration_list=['2000','3500'],
    is_init_state=False,
    init_state=[0.1,0.1,0.5,0.01,0],
    save_render=False,
    legend_list=[ 'INFADP-2000','INFADP-3500'],
    use_opt=False,
    constrained_env=False,
    is_tracking=True,
    dt=None)

runer.run()

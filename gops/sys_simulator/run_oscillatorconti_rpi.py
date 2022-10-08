from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/RPI/221008-160556"] * 2,
    trained_policy_iteration_list=['40', '50'],
    is_init_state=True,
    init_state=[1, -1],
    save_render=False,
    legend_list=['RPI-40', 'RPI-50'],
    use_opt=False,
    constrained_env=False,
    is_tracking=False,
    dt=None)

runer.run()

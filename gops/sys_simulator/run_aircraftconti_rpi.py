from sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["../../results/RPI/poly_aircraftconti_221017-233026"] * 2,
    trained_policy_iteration_list=['40', '50'],
    is_init_info=True,
    init_info={'init_state':[0.3, -0.5, 0.2]},
    save_render=False,
    legend_list=['RPI-40', 'RPI-50'],
    use_opt=True,
    constrained_env=False,
    is_tracking=False,
    # obs_noise_data = [[-0.01,0,-0.01],[0.01,0,0.01]],
    # obs_noise_type = 'uniform',
    # action_noise_data=[[0],[0.01]],
    # action_noise_type='normal',

    dt=None)

runner.run()

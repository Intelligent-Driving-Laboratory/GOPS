from sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["../../results/RPI/poly_suspension_221116-234126"] * 2,
    trained_policy_iteration_list=['40', '50'],
    is_init_info=True,
    init_info={'init_state': [0.0, 0.0, 0.0, 0.0]},  # [0.02, 0.3, 0.08, -0.6]
    save_render=False,
    legend_list=['RPI-40', 'RPI-50'],
    use_opt=False,
    constrained_env=False,
    use_dist=True,
    is_tracking=False,
    dt=None)

runner.run()

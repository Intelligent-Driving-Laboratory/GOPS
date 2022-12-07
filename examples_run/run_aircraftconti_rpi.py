from gops.sys_simulator.sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["../../results/RPI/poly_aircraftconti_221116-233344"] * 2,
    trained_policy_iteration_list=["40", "50"],
    is_init_info=True,
    init_info={"init_state": [0.3, -0.5, 0.2]},
    save_render=False,
    legend_list=["RPI-40", "RPI-50"],
    use_opt=True,
    constrained_env=False,
    is_tracking=False,
    opt_args={"opt_controller_type": "OPT"},
    dt=None,
)

runner.run()

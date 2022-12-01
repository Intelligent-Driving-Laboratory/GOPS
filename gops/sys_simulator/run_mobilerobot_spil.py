from sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["../../results/SAC/221201-005438"]*1,
    trained_policy_iteration_list=['34800_opt'],
    is_init_info=True,
    init_info={"init_state": [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.5, -4., 1.57, 0.4, 1.57]},
    save_render=False,
    legend_list=['9000'],
    use_opt=False,
    # plot_range= [0,50],
    constrained_env=True,
    opt_args={
        "opt_controller_type": "MPC",
        "num_pred_step": 25,
        "gamma": 0.99,
        "minimize_options": {
            "max_iter": 200,
            "tol": 1e-3,
            "acceptable_tol": 1e0,
            "acceptable_iter": 10,
            # "print_level": 5,
        },
        "use_terminal_cost": False,
        # "terminal_cost": terminal_cost,
    }
)

runner.run()

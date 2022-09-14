from policy_runner import PolicyRuner


runer = PolicyRuner(
    log_policy_dir_list=["../../results/PPO/220914-124352"],
    trained_policy_iteration_list=[100],
    save_render=False,
    init_state=[1.0, 1.0, 1.0])

runer.run()
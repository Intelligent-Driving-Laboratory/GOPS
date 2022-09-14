from policy_runner import PolicyRuner


runer = PolicyRuner(
    log_policy_dir_list=["../../results/SAC/220914-134325"],
    trained_policy_iteration_list=[12000],
    save_render=False,)

runer.run()
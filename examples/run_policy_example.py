from gops.trainer.policy_runner import PolicyRuner


runer = PolicyRuner(
    log_policy_dir_list=["../results/DDPG/220815-153424"],
    trained_policy_iteration_list=[10_000],
    save_render=True)

runer.run()
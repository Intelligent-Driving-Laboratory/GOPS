from sys_run import PolicyRunner

runner = PolicyRunner(
    log_policy_dir_list=["./results/DDPG/simu_lqs2a1"]*2,
    trained_policy_iteration_list=['250000', '220000'],
    is_init_info=True,
    init_info={"init_state":[0.5, -0.5]},
    save_render=False,
    legend_list=['DDPG-250000', 'DDPG-220000'],
    use_opt=True)

runner.run()

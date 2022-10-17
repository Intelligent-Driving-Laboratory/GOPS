from sys_run import PolicyRuner

runer = PolicyRuner(
    log_policy_dir_list=["../../results/FHADP/221017-195545"]*1,
    trained_policy_iteration_list=['4000'],
    is_init_info=True,
    init_info={"init_state":[1, 1]},
    save_render=False,
    legend_list=['4000'],
    plot_range=[0,200],
    dt=0.05,
    use_opt=True)

runer.run()

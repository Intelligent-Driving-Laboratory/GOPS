from sys_run import PolicyRuner

def controller_func(x):
    u = -x+4
    return u
runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/220919-104858"],
    trained_policy_iteration_list=['4000'],
    init_state=[2,-0.1],
    save_render=False)

runer.run()

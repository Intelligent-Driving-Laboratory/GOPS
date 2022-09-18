from policy_runner import PolicyRuner

def controller_func(x):
    u = -x+4
    return u
runer = PolicyRuner(
    log_policy_dir_list=["../../results/INFADP/lqr_s5a1_wo_reward_scale"
                         ,"../../results/INFADP/lqr_s5a1_wt_reward_scale"],
    trained_policy_iteration_list=['6400','6400'],
    init_state=[0.2,-0.1,-0.05,0.0,0.03],
    save_render=False)

runer.run()

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#  Creator: iDLab
#  Description: check the characteristic of dynamic system, raise warning if there are potential problems


from json import load
import numpy as np
import warnings
import os
import argparse
import torch
from gops.utils.common_utils import get_args_from_json

from gops.utils.plot_evaluation import self_plot
# from gops.sys_simulator.sys_run import default_cfg
# import matplotlib.pyplot as plt
# import seaborn as sns


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s: %s\n' % (category.__name__, message)

def load_args(log_policy_dir):
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args

def load_policy(log_policy_dir, trained_policy_iteration):
    args = load_args(log_policy_dir)
    # Create policy
    alg_name = args["algorithm"]
    alg_file_name = alg_name.lower()
    file = __import__(alg_file_name)
    ApproxContainer = getattr(file, "ApproxContainer")
    networks = ApproxContainer(**args)
    # Load trained policy
    log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(trained_policy_iteration)
    networks.load_state_dict(torch.load(log_path))
    return networks

def check_dynamic(env):
    """
    check whether the dynamic characteristic is well behaved.

    """

    # First, calculate the first/second-order difference of state
    df_state_list = []
    ddf_state_list = []

    stable_final = np.ones(env.work_space.shape[1]).astype(np.bool_)
    stable_final_threshold = (env.work_space[1] - env.work_space[0]) * 0.0001

    # close-loop if needed
    network = load_policy("results/SAC/idp_221017-174348", '24000')

    for simu_round in range(1):
        env.reset()
        # state = env.obs
        state = env.state
        done = False

        for step_index in range(20010):
            # action = np.zeros(env.action_space.sample().shape)
            action = network.policy(torch.tensor(state,dtype=torch.float)).detach_().numpy()
            _, _, done, info = env.step(action)

            if done:
                # record stability beyond time limit
                if ("TimeLimit.truncated" in info) and info["TimeLimit.truncated"]:
                    stable_final = stable_final & (df_state_list[-1][0] < stable_final_threshold)
                break
                
            state_next = env.obs # env.state()
            df_state_list.append((state_next - state, state))

            if step_index > 0:
                ddf_state_list.append((df_state_list[-1][0] - df_state_list[-2][0], df_state_list[-1][0], state))

            state = state_next
    
    # Second, raise warning if dynamic change too fast
    df_threshold = (env.work_space[1] - env.work_space[0]) * 0.4
    df_too_fast = np.zeros(env.work_space.shape[1]).astype(np.bool_)
    for i in range(len(df_state_list)):
        df_state, _ = df_state_list[i]
        df_too_fast = df_too_fast | (df_state > df_threshold)
    
    ddf_threshold = (env.work_space[1] - env.work_space[0]) * 0.3
    ddf_too_fast = np.zeros(env.work_space.shape[1]).astype(np.bool_)
    for i in range(len(ddf_state_list)):
        ddf_state, _, _ = ddf_state_list[i]
        ddf_too_fast = ddf_too_fast | (ddf_state > ddf_threshold)
    
    # print(len(df_state_list))
    
    warnings.filterwarnings("always")
    origin_format = warnings.formatwarning
    warnings.formatwarning = warning_on_one_line
    for i in range(env.work_space.shape[1]):
        if df_too_fast[i] or ddf_too_fast[i]:
            warnings.warn("the %d-th state changes too fast, please check the environment dynamic. If you ensure there is no problems, please ignore the warning." % i)

    # Third, raise warning if the open-loop system not stabilizes beyond time limit
    for i in range(env.work_space.shape[1]):
        if not stable_final[i]:
            warnings.warn("the %d-th state can not stablize without action, please check the environment. If you ensure there is no problems, please ignore the warning." % i)
    warnings.formatwarning = origin_format
    warnings.filterwarnings("ignore")

    # Fourth, draw df_state and ddf_state distribution figure
    for i in range(env.work_space.shape[1]):
        data = [dict({"x":x[i], "y":y[i]}) for y, x in df_state_list]
        self_plot(data, category='scatter', fname="state_%d_first_order_difference" % i, xlabel='State-%d'%i, ylabel='First-order difference')

        data = [dict({"x":x[i], "y":y[i]}) for y, _, x in ddf_state_list]
        self_plot(data, category='scatter', fname="state_%d_second_order_difference" % i, xlabel='State-%d'%i, ylabel='Second-order difference')

        data = [dict({"x":x[i], "y":y[i]}) for y, x, _ in ddf_state_list]
        self_plot(data, category='scatter', fname="state_%d_difference_distribution" % i, xlabel='First-order difference of state-%d'%i, ylabel='Second-order difference of state-%d'%i)

    print("Complete the check of environment dynamic.")

if __name__ == "__main__":

    from gops.create_pkg.create_env import create_env
    # env = create_env(env_id='pyth_lq', lq_config='s4a2')

    env = create_env(env_id='pyth_idpendulum')

    check_dynamic(env)

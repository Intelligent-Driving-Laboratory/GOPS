#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
# 
#  Creator: Xujie Song
#  Description: check the characteristic of dynamic system,
#               draw the figures of first/second-order difference of state,
#               raise warning if there are potential problems

import numpy as np
import warnings
import os
import copy
import json
import datetime
import argparse
import torch
from itertools import cycle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from gops.utils.common_utils import change_type, get_args_from_json
from gops.utils.plot_evaluation import cm2inch
from gops.create_pkg.create_env import create_env


default_cfg = dict()
default_cfg["fig_size"] = (28, 21)
default_cfg["dpi"] = 300
default_cfg["pad"] = 1

default_cfg["tick_size"] = 8
default_cfg["tick_label_font"] = "Times New Roman"
default_cfg["legend_font"] = {
    "family": "Times New Roman",
    "size": "12",
    "weight": "normal",
}
default_cfg["label_font"] = {
    "family": "Times New Roman",
    "size": "13",
    "weight": "normal",
}

default_cfg["img_fmt"] = "png"

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s: %s\n' % (category.__name__, message)

def load_args(log_policy_dir):
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args

def load_policy(args, policy_path):
    # Create policy
    alg_name = args["algorithm"]
    alg_file_name = alg_name.lower()
    file = __import__(alg_file_name)
    ApproxContainer = getattr(file, "ApproxContainer")
    networks = ApproxContainer(**args)
    # Load trained policy
    networks.load_state_dict(torch.load(policy_path))
    return networks

def compute_action(obs, networks):
    batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
    logits = networks.policy(batch_obs)
    action_distribution = networks.create_action_distributions(logits)
    action = action_distribution.mode()
    action = action.detach().numpy()[0]
    return action

def draw_figures(traj_data, df_data, ddf_data, ddf_df_data, state_index, close_loop=False, save_path=None):

    traj_num = len(traj_data)

    # color list
    tableau_colors = cycle(mcolors.TABLEAU_COLORS)
    color_list = [next(tableau_colors) for _ in range(traj_num)]

    # sub-figure
    fig, ax = plt.subplots(2, 2, figsize=cm2inch(*default_cfg["fig_size"]), dpi=default_cfg["dpi"])
    if close_loop:
        fig.suptitle("Close-loop Dynamic Check of State-%d"%state_index)
    else:
        fig.suptitle("Open-loop Dynamic Check of State-%d"%state_index)
    
    # plot traj
    for i in range(traj_num):
        ax[0,0].plot(traj_data[i]["x"], traj_data[i]["y"], color=color_list[i])
    ax[0,0].set_xlabel('Time Step', default_cfg["label_font"])
    ax[0,0].set_ylabel('State-%d'%state_index, default_cfg["label_font"])
    ax[0,0].tick_params(labelsize=default_cfg["tick_size"])
    # ax[0,0].legend(['episode %d'%i for i in range(1, traj_num+1)],
    #                loc='best', ncol=1, prop=default_cfg["legend_font"])

    # plot df_data
    for i in range(traj_num):
        ax[0,1].scatter(df_data[i]["x"], df_data[i]["y"], s=7, color=color_list[i])
    ax[0,1].set_xlabel('State-%d'%state_index, default_cfg["label_font"])
    ax[0,1].set_ylabel(r'$\Delta$(State-%d)'%state_index, default_cfg["label_font"])
    # ax[0,1].legend(['episode %d'%i for i in range(1, traj_num+1)],
    #                loc='best', ncol=1, prop=default_cfg["legend_font"])

    # plot ddf_data
    for i in range(traj_num):
        ax[1,0].scatter(ddf_data[i]["x"], ddf_data[i]["y"], s=7, color=color_list[i])
    ax[1,0].set_xlabel('State-%d'%state_index, default_cfg["label_font"])
    ax[1,0].set_ylabel(r'$\Delta^2$(State-%d)'%state_index, default_cfg["label_font"])
    # ax[1,0].legend(['episode %d'%i for i in range(1, traj_num+1)],
    #                loc='best', ncol=1, prop=default_cfg["legend_font"])

    # plot df_ddf
    for i in range(traj_num):
        ax[1,1].scatter(ddf_df_data[i]["x"], ddf_df_data[i]["y"], s=7, color=color_list[i])
    ax[1,1].set_xlabel(r'$\Delta$(State-%d)'%state_index, default_cfg["label_font"])
    ax[1,1].set_ylabel(r'$\Delta^2$(State-%d)'%state_index, default_cfg["label_font"])
    # ax[1,1].legend(['episode %d'%i for i in range(1, traj_num+1)],
    #                loc='best', ncol=1, prop=default_cfg["legend_font"])
    
    # config layout
    fig.tight_layout(pad=default_cfg["pad"])

    # save fig
    if save_path is None:
        save_path = os.path.join('./')
    if close_loop:
        fig.savefig(os.path.join(save_path, 'Close-loop Dynamic Check of State-%d.png'%state_index))
    else:
        fig.savefig(os.path.join(save_path, 'Open-loop Dynamic Check of State-%d.png'%state_index))
    

def check_dynamic(env=None, traj_num=5, init_info=None, log_policy_dir=None, policy_iteration=None):
    """
    check whether the dynamic characteristic is well behaved.

    """

    # close-loop if needed
    close_loop = False
    if log_policy_dir is not None:
        close_loop = True
        # recreate env
        has_env = (env is not None)
        args = load_args(log_policy_dir)
        env = create_env(**args)
        if has_env:
            print("The env is recreated successfully according to \'config.json\' in \'%s\'."% log_policy_dir)
        # load policy
        policy_path = os.path.join(log_policy_dir, 'apprfunc', "apprfunc_{}.pkl".format(policy_iteration))
        controller = load_policy(args, policy_path)
    
    assert env is not None

    save_path = os.path.join('./', 'figures')
    if hasattr(env, 'config_name'):
        env_name = '%s_%s'%(env.env_id, env.config_name)
    else:
        env_name = env.env_id
    
    if close_loop:
        save_path = os.path.join(save_path, '%s_close_test'%env_name)
    else:
        save_path = os.path.join(save_path, '%s_open_test'%env_name)
    save_path = os.path.join(save_path, datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    os.makedirs(save_path, exist_ok=True)

    args = {'env_name':env_name,
            'traj_num':traj_num,
            'init_info':init_info,
            'log_policy_dir':log_policy_dir,
            'policy_iteration':policy_iteration}
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(change_type(copy.deepcopy(args)), f, ensure_ascii=False, indent=4)

    if init_info is not None:
        assert len(init_info["init_state"]) == traj_num

    # First, calculate the first/second-order difference of state
    state_list = []
    df_state_list = []
    ddf_state_list = []

    if hasattr(env, 'work_space'):
        state_dim = env.work_space.shape[1]
        state_range = env.work_space[1] - env.work_space[0]
    else:
        state_dim = env.observation_space.shape[0]
        state_range = 1 # the range is usually [-inf, inf] for gym, so just give 1 here for simplicity
    
    stable_final = np.ones(state_dim).astype(np.bool_)
    stable_final_threshold = state_range * 0.0001

    for episode in range(traj_num):
        if init_info is not None:
            init_info_episode = {k: init_info[k][episode] for k in init_info}
            obs, info = env.reset(**init_info_episode)
        else:
            obs, info = env.reset()
        state_list.append([])
        df_state_list.append([])
        ddf_state_list.append([])

        state = env.state
        done = False

        for step_index in range(20010):
            state_list[-1].append((obs, step_index))

            if close_loop:
                action = compute_action(obs, controller)
            else:
                action = np.zeros(env.action_space.sample().shape)
            obs_next, _, done, info = env.step(action)

            # record stability beyond time limit
            if ("TimeLimit.truncated" in info) and info["TimeLimit.truncated"]:
                stable_final = stable_final & (df_state_list[-1][-1][0] < stable_final_threshold)
                break

            if done:
                break
                
            state_next = env.state

            df_state_list[-1].append((obs_next - obs, obs))

            if step_index > 0:
                ddf_state_list[-1].append((df_state_list[-1][-1][0] - df_state_list[-1][-2][0], df_state_list[-1][-1][0], obs))

            state = state_next
            obs = obs_next

    # Second, raise warning if dynamic change too fast
    df_threshold = state_range * 0.4
    df_too_fast = np.zeros(state_dim).astype(np.bool_)
    for k in range(len(df_state_list)):
        for i in range(len(df_state_list[k])):
            df_state, _ = df_state_list[k][i]
            df_too_fast = df_too_fast | (df_state > df_threshold)
        
    ddf_threshold = state_range * 0.3
    ddf_too_fast = np.zeros(state_dim).astype(np.bool_)
    for k in range(len(df_state_list)):
        for i in range(len(ddf_state_list[k])):
            ddf_state, _, _ = ddf_state_list[k][i]
            ddf_too_fast = ddf_too_fast | (ddf_state > ddf_threshold)
    
    warnings.filterwarnings("always")
    origin_format = warnings.formatwarning
    warnings.formatwarning = warning_on_one_line
    for i in range(state_dim):
        if df_too_fast[i] or ddf_too_fast[i]:
            warnings.warn("the %d-th state changes too fast, please check the environment dynamic. If you ensure there is no problems, please ignore the warning." % i)
    '''
    # Third, raise warning if the open-loop system not stabilizes beyond time limit
    for i in range(state_dim):
        if not stable_final[i]:
            warnings.warn("the %d-th state can not stablize without action, please check the environment. If you ensure there is no problems, please ignore the warning." % i)
    '''
    warnings.formatwarning = origin_format
    warnings.filterwarnings("ignore")

    # Fourth, draw df_state and ddf_state distribution figures with trajectory figures
    for i in range(state_dim):
        traj_data = [dict({"x":[x for y, x in state_list[k]],\
                          "y":[y[i] for y, x in state_list[k]]})
                     for k in range(traj_num)]

        df_data = [dict({"x":[x[i] for y, x in df_state_list[k]],\
                        "y":[y[i] for y, x in df_state_list[k]]})
                   for k in range(traj_num)]

        ddf_data = [dict({"x":[x[i] for y, _, x in ddf_state_list[k]],\
                         "y":[y[i] for y, _, x in ddf_state_list[k]]})
                    for k in range(traj_num)]

        ddf_df_data = [dict({"x":[x[i] for y, x, _ in ddf_state_list[k]],\
                            "y":[y[i] for y, x, _ in ddf_state_list[k]]})
                       for k in range(traj_num)]

        draw_figures(traj_data, df_data, ddf_data, ddf_df_data, state_index=i+1, close_loop=close_loop, save_path=save_path)

    print("Complete the check of environment dynamic.")

if __name__ == "__main__":
    
    '''
    Here are some examples, including:
    (1.a) open-loop check of pyth_lqs2a1_data
    (1.b) close-loop check of pyth_lqs2a1_data

    (2.a) open-loop check of pyth_lqs4a2_data
    (2.b) close-loop check of pyth_lqs4a2_data
    
    (3.a) open-loop check of simu_lqs2a1conti_data
    (3.b) close-loop check of simu_lqs2a1conti_data
    
    (4.a) open-loop check of pyth_idpendulum_data
    (4.b) close-loop check of pyth_idpendulum_data
    
    (5.a) open-loop check of pyth_aircraftconti
    (5.b) close-loop check of pyth_aircraftconti
    
    (6.a) open-loop check of pyth_veh2dofconti
    (6.b) close-loop check of pyth_veh2dofconti

    (7.a) open-loop check of pyth_veh3dofconti
    (7.b) close-loop check of pyth_veh3dofconti

    (8.a) open-loop check of gym_pendulum
    (8.b) close-loop check of gym_pendulum
    '''

    '''(1.a) open-loop check of pyth_lqs2a1_data'''
    env = create_env(env_id='pyth_lq', lq_config='s2a1')
    check_dynamic(env, traj_num=2,
                  init_info={"init_state": [[0.5, -0.5], [1, -1]]})
    
    '''(1.b) close-loop check of pyth_lqs2a1_data'''
    # env = create_env(env_id='pyth_lq', lq_config='s2a1')
    # check_dynamic(env,
    #               log_policy_dir='./results/INFADP/s2a1',
    #               policy_iteration='350000')

    '''(2.a) open-loop check of pyth_lqs4a2_data'''
    # env = create_env(env_id='pyth_lq', lq_config='s4a2')
    # check_dynamic(env)
    
    '''(2.b) close-loop check of pyth_lqs4a2_data'''
    # env = create_env(env_id='pyth_lq', lq_config='s4a2')
    # check_dynamic(env, traj_num=3, 
    #               log_policy_dir='./results/INFADP/s4a2',
    #               policy_iteration='100000')

    '''(3.a) open-loop check of simu_lqs2a1conti_data'''
    # env = create_env(**load_args('./results/DDPG/221102-220805'))
    # check_dynamic(env)

    '''(3.b) close-loop check of simu_lqs2a1conti_data'''
    # env = create_env(**load_args('./results/DDPG/simu_lqs2a1'))
    # check_dynamic(env, traj_num=3,
    #               init_info={"init_state": [[0.5, -0.3], [1, -1], [-1, 0.7]]},
    #               log_policy_dir='./results/DDPG/simu_lqs2a1',
    #               policy_iteration='250000')

    '''(4.a) open-loop check of pyth_idpendulum_data'''
    # env = create_env(env_id='pyth_idpendulum')
    # check_dynamic(env)

    '''(4.b) close-loop check of pyth_idpendulum_data'''
    # env = create_env(env_id='pyth_idpendulum')
    # check_dynamic(env, traj_num=5,
    #               log_policy_dir='./results/SAC/idp_221017-174348',
    #               policy_iteration='27000')

    '''(5.a) open-loop check of pyth_aircraftconti'''
    # env = create_env(env_id='pyth_aircraftconti', is_adversary=True,
    #                  gamma_atte=5, state_threshold=[2.0, 2.0, 2.0], max_episode_steps=200)
    # check_dynamic(env)

    '''(5.b) close-loop check of pyth_aircraftconti'''
    # env = create_env(env_id='pyth_aircraftconti', is_adversary=True,
    #                  gamma_atte=5, state_threshold=[2.0, 2.0, 2.0], max_episode_steps=200)
    # check_dynamic(env, traj_num=1, init_info={'init_state':[[0.3, -0.5, 0.2]]},
    #               log_policy_dir='./results/RPI/poly_aircraftconti_221107-195932',
    #               policy_iteration='50')

    '''(6.a) open-loop check of pyth_veh2dofconti'''
    # env = create_env(env_id='pyth_veh2dofconti', pre_horizon=10)
    # check_dynamic(env, traj_num=10)

    '''(6.b) close-loop check of pyth_veh2dofconti'''
    # env = create_env(env_id='pyth_veh2dofconti', pre_horizon=10)
    # check_dynamic(env, traj_num=1, init_info={"init_state": [[1., 0., 0., 0.]], "ref_time": [0.], "ref_num": [0.]},
    #               log_policy_dir='./results/INFADP/veh2dofconti_221017-211644',
    #               policy_iteration='3000')

    '''(7.a) open-loop check of pyth_veh3dofconti'''
    # env = create_env(env_id='pyth_veh3dofconti', pre_horizon=10)
    # check_dynamic(env, traj_num=10)

    '''(7.b) close-loop check of pyth_veh3dofconti'''
    # env = create_env(env_id='pyth_veh3dofconti', pre_horizon=10)
    # check_dynamic(env, traj_num=1, init_info={"init_state": [[0., 0., 0., 10., 0, 0]], "ref_time": [0.], "ref_num": [0.]},
    #               log_policy_dir='./results/INFADP/veh3dofconti_221017-210557',
    #               policy_iteration='4000')

    '''(8.a) open-loop check of gym_pendulum'''
    # env = create_env(env_id='gym_pendulum')
    # check_dynamic(env, traj_num=2)

    '''(8.b) close-loop check of gym_pendulum'''
    # env = create_env(env_id='gym_pendulum')
    # check_dynamic(env, traj_num=2,
    #               log_policy_dir='./results/DDPG/gym_pendulum',
    #               policy_iteration='8000')
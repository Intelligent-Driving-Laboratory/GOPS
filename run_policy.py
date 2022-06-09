import argparse
import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import torch
from gym import wrappers
from time import time       # just to have timestamps in the files

from gops.create_pkg.create_env import create_env





def get_args_from_json(json_file_path, args_dict):
    import json
    summary_filename = json_file_path
    with open(summary_filename) as f:
        summary_dict = json.load(fp=f)

    for key in summary_dict.keys():
        args_dict[key] = summary_dict[key]

    return args_dict

def run_an_episode(env, networks, init_state, render=True):

    obs_list = []
    action_list = []
    reward_list = []
    step = 0
    step_list = []
    obs = env.reset()
    if len(init_state) == 0:
        pass
    elif len(obs) == len(init_state):
        obs = np.array(init_state)
    else:
        raise NotImplementedError("The dimension of Initial state is wrong!")
    done = 0
    info = {"TimeLimit.truncated": False}
    while not (done or info["TimeLimit.truncated"]):
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = networks.policy(batch_obs)
        action_distribution = networks.create_action_distributions(logits)
        action = action_distribution.mode()
        action = action.detach().numpy()[0]
        step_list.append(step)
        next_obs, reward, done, info = env.step(action)
        step = step + 1
        obs_list.append(obs)
        action_list.append(action)
        obs = next_obs
        if "TimeLimit.truncated" not in info.keys():
            info["TimeLimit.truncated"] = False
        # Draw environment animation
        if render:
            env.render()
        reward_list.append(reward)

    eval_dict = {
        "reward_list": reward_list,
        "action_list": action_list,
        "obs_list": obs_list,
        "step_list": step_list,
    }

    return eval_dict

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)

def draw_save(eval_list, algorithm_list, env_id, plot_range):
    # create save directory
    import os
    path = 'policy_result'
    algs_name = ""
    for item in algorithm_list:
        algs_name = algs_name + item + '-'
    path = os.path.join(path, algs_name + env_id)
    os.makedirs(path, exist_ok=True)

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    default_cfg = dict()

    default_cfg["fig_size"] = (12, 9)
    default_cfg["dpi"] = 300
    default_cfg["pad"] = 0.5

    default_cfg["tick_size"] = 8
    default_cfg["tick_label_font"] = "Times New Roman"
    default_cfg["legend_font"] = {
        "family": "Times New Roman",
        "size": "8",
        "weight": "normal",
    }
    default_cfg["label_font"] = {
        "family": "Times New Roman",
        "size": "9",
        "weight": "normal",
    }

    fig_size = (
        default_cfg["fig_size"],
        default_cfg["fig_size"],
    )

    action_dim = eval_list[0]["action_list"][0].shape[0]
    obs_dim = eval_list[0]["obs_list"][0].shape[0]
    data_lenth = len(eval_list[0]["step_list"])
    policy_num = len(algorithm_list)



    # Create initial array
    reward_array = np.zeros((data_lenth, 1 * policy_num))
    action_array = np.zeros((data_lenth, action_dim * policy_num))
    state_array = np.zeros((data_lenth, obs_dim * policy_num))

    # Put data into array
    for i in range(policy_num):
        reward_array[:, i] = np.array(eval_list[i]["reward_list"])
        action_array[:, i*action_dim:(i+1)*action_dim] = np.array(eval_list[i]["action_list"])
        state_array[:, i*obs_dim:(i+1)*obs_dim] = np.array(eval_list[i]["obs_list"])
    step_array = np.array(eval_list[0]["step_list"])

    if len(plot_range) == 0:
        pass
    elif len(plot_range) == 2:
        if len(step_array) <= plot_range[0]:
            raise NotImplementedError("The setting of plot range is out of range")
        else:
            reward_array = reward_array[plot_range[0]:plot_range[1], :]
            action_array = action_array[plot_range[0]:plot_range[1], :]
            state_array = state_array[plot_range[0]:plot_range[1], :]
            step_array = step_array[plot_range[0]:plot_range[1]]
    else:
        raise NotImplementedError("The setting of plot range is wrong")

    # plot reward
    path_reward = os.path.join(path, 'reward')
    path_reward_tiff = os.path.join(path_reward, 'reward.tiff')
    os.makedirs(path_reward, exist_ok=True)

    fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
    for i in range(policy_num):
        sns.lineplot(x=step_array, y=reward_array[:, i], label='{}'.format(algorithm_list[i]))
    plt.tick_params(labelsize=default_cfg["tick_size"])
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
    plt.xlabel('Time step', default_cfg["label_font"])
    plt.ylabel('Reward', default_cfg["label_font"])
    plt.legend(loc='best', prop=default_cfg["legend_font"])
    fig.tight_layout(pad=default_cfg["pad"])
    plt.savefig(path_reward_tiff, format='tiff', bbox_inches='tight')
    # plt.show()

    # plot action
    path_action = os.path.join(path, 'action')
    os.makedirs(path_action, exist_ok=True)

    for j in range(action_dim):
        path_action_tiff = os.path.join(path_action, 'action-{}.tiff'.format(j))
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
        for i in range(policy_num):
            sns.lineplot(x=step_array, y=action_array[:, j + i * action_dim], label='{}'.format(algorithm_list[i]))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel('Time step', default_cfg["label_font"])
        plt.ylabel('Action-{}'.format(j), default_cfg["label_font"])
        plt.legend(loc='best', prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_action_tiff, format='tiff', bbox_inches='tight')
        # plt.show()

    # plot state
    path_state = os.path.join(path, 'state')
    os.makedirs(path_state, exist_ok=True)

    for j in range(obs_dim):
        path_state_tiff = os.path.join(path_state, 'state-{}.tiff'.format(j))
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
        for i in range(policy_num):
            sns.lineplot(x=step_array, y=state_array[:, j+i*obs_dim], label='{}'.format(algorithm_list[i]))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel('Time step', default_cfg["label_font"])
        plt.ylabel('State_{}'.format(j), default_cfg["label_font"])
        plt.legend(loc='best', prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_state_tiff, format='tiff', bbox_inches='tight')
        # plt.show()



if __name__ == "__main__":

    # Key Parameters for users
    ######################################################
    # log_policy_dir_list = ["results/DDPG/220509-185827"
    #                        ]

    log_policy_dir_list = ["results/SAC/220509-182716",
                           "results/DDPG/220509-185827"]
    trained_policy_iteration_list = [5000, 5000]
    plot_range = []
    init_state = []
    save_render = False
    #####################################################

    eval_list = []
    env_ids = []
    algorithm_list = []
    policy_num = len(log_policy_dir_list)
    if policy_num != len(trained_policy_iteration_list):
        raise NotImplementedError("The lenth of policy number is not equal to the number of policy iteration")
    for i in range(policy_num):
        log_policy_dir = log_policy_dir_list[i]
        json_path = log_policy_dir + "/config.json"

        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)

        # Create environment
        env = create_env(**args)
        if save_render:
            env = wrappers.Monitor(env, './videos/' + str(args["algorithm"]) + '/')
        args["action_high_limit"] = env.action_space.high
        args["action_low_limit"] = env.action_space.low

        # Create policy
        alg_name = args["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        networks = ApproxContainer(**args)
        print("Create {}-policy successfully!".format(alg_name))

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(trained_policy_iteration_list[i])
        networks.load_state_dict(torch.load(log_path))
        print("Load {}-policy successfully!".format(alg_name))

        # Run policy
        eval_dict = run_an_episode(env, networks, init_state, render=False)
        eval_list.append(eval_dict)

        env_id = args["env_id"]
        env_ids.append(env_id)
        algorithm_list.append(alg_name)

    # Plot and save
    if len(env_ids) > 1:
        for i in range(len(env_ids)-1):
            if env_ids[i] != env_ids[i+1]:
                raise NotImplementedError("policy {} and policy {} is not trained in the same environment".format(i, i+1))
    env_id = env_ids[0]
    draw_save(eval_list, algorithm_list, env_id, plot_range)


import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from gym import wrappers

from gops.create_pkg.create_env import create_env
from gops.utils.plot import cm2inch
from gops.utils.utils import get_args_from_json, mp4togif

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

default_cfg["img_fmt"] = "png"


class PolicyRuner():
    def __init__(self, log_policy_dir_list, trained_policy_iteration_list, save_render=False, plot_range=[],
                 init_state=[]) -> None:
        self.log_policy_dir_list = log_policy_dir_list
        self.trained_policy_iteration_list = trained_policy_iteration_list
        self.save_render = save_render
        self.args = None
        self.plot_range = plot_range
        self.init_state = init_state
        self.policy_num = len(self.log_policy_dir_list)
        if self.policy_num != len(self.trained_policy_iteration_list):
            raise RuntimeError("The lenth of policy number is not equal to the number of policy iteration")

        # data for plot

        #####################################################
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()
        # save path

        path = os.path.join(os.path.dirname(__file__), "..", "..", "policy_result")
        path = os.path.abspath(path)

        algs_name = ""
        for item in self.algorithm_list:
            algs_name = algs_name + item + "-"
        self.save_path = os.path.join(path, algs_name + self.env_id, datetime.datetime.now().strftime("%y%m%d-%H%M%S"))

        os.makedirs(self.save_path, exist_ok=True)

    def run_an_episode(self, env, networks, init_state, render=True):
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
            # K =np.array([11.5510,11.0562,2.8245,-9.7830,-8.9332])
            # action = -np.array(np.dot(K,obs)).reshape(-1)
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

    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        obs_dim = self.eval_list[0]["obs_list"][0].shape[0]
        data_lenth = len(self.eval_list[0]["step_list"])
        policy_num = len(self.algorithm_list)

        # Create initial array

        reward_array_list = []
        action_array_list = []
        state_array_list = []
        step_array_list = []
        # Put data into array
        for i in range(policy_num):
            reward_array_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_array_list.append(np.array(self.eval_list[i]["action_list"]))
            state_array_list.append(np.array(self.eval_list[i]["obs_list"]))
            step_array_list.append(np.array(self.eval_list[i]["step_list"]))

        if len(self.plot_range) == 0:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = np.min(self.plot_range[1], reward_array_list[i].shape[0])

                reward_array_list[i] = reward_array_list[i][start_range: end_range]
                action_array_list[i] = action_array_list[i][start_range: end_range]
                state_array_list[i] = state_array_list[i][start_range: end_range]
                step_array_list[i] = step_array_list[i][start_range: end_range]
        else:
            raise NotImplementedError("The setting of plot range is wrong")

        # plot reward
        path_reward = os.path.join(self.save_path, "reward")
        path_reward_fmt = os.path.join(path_reward, "reward.{}".format(default_cfg["img_fmt"]))
        os.makedirs(path_reward, exist_ok=True)

        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
        for i in range(policy_num):
            print("=====")
            print(step_array_list[i].shape, reward_array_list[i].shape)
            sns.lineplot(x=step_array_list[i], y=reward_array_list[i], label="{}".format(self.algorithm_list[i]))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel("Time step", default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        # plt.show()

        # plot action
        path_action = os.path.join(self.save_path, "action")
        os.makedirs(path_action, exist_ok=True)

        for j in range(action_dim):
            path_action_fmt = os.path.join(path_action, "action-{}.{}".format(j, default_cfg["img_fmt"]))
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
            for i in range(policy_num):
                sns.lineplot(x=step_array_list[i], y=action_array_list[i][:, j],
                             label="{}".format(self.algorithm_list[i]))
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel("Time step", default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
            # plt.show()

        # plot state
        path_state = os.path.join(self.save_path, "state")
        os.makedirs(path_state, exist_ok=True)

        for j in range(obs_dim):
            path_state_fmt = os.path.join(path_state, "state-{}.{}".format(j, default_cfg["img_fmt"]))
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
            for i in range(policy_num):
                sns.lineplot(x=step_array_list[i], y=state_array_list[i][:, j],
                             label="{}".format(self.algorithm_list[i]))
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel("Time step", default_cfg["label_font"])
            plt.ylabel("State_{}".format(j), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")
        # plt.show()

    @staticmethod
    def __load_args(log_policy_dir):
        json_path = os.path.join(log_policy_dir, "config.json")
        parser = argparse.ArgumentParser()
        args_dict = vars(parser.parse_args())
        args = get_args_from_json(json_path, args_dict)
        return args

    def __load_all_args(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            args = self.__load_args(log_policy_dir)
            self.args_list.append(args)
            env_id = args["env_id"]
            self.env_id_list.append(env_id)
            self.algorithm_list.append(args["algorithm"])

    def __load_env(self):
        env = create_env(**self.args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            env = wrappers.RecordVideo(env, video_path,
                                       name_prefix="{}_video".format(self.args["algorithm"]))
        self.args["action_high_limit"] = env.action_space.high
        self.args["action_low_limit"] = env.action_space.low
        return env

    def __load_policy(self, log_policy_dir, trained_policy_iteration):
        # Create policy
        alg_name = self.args["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        networks = ApproxContainer(**self.args)
        print("Create {}-policy successfully!".format(alg_name))

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(trained_policy_iteration)
        networks.load_state_dict(torch.load(log_path))
        print("Load {}-policy successfully!".format(alg_name))
        return networks

    def __run_data(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            trained_policy_iteration = self.trained_policy_iteration_list[i]

            self.args = self.args_list[i]
            env = self.__load_env()
            networks = self.__load_policy(log_policy_dir, trained_policy_iteration)

            # Run policy
            eval_dict = self.run_an_episode(env, networks, self.init_state, render=False)
            # mp4 to gif
            self.eval_list.append(eval_dict)

    def __save_mp4_as_gif(self):
        if self.save_render:
            videos_path = os.path.join(self.save_path, "videos")

            videos_list = [i for i in glob.glob(os.path.join(videos_path, "*.mp4"))]
            for v in videos_list:
                mp4togif(v)

    def get_n_verify_env_id(self):
        env_id = self.env_id_list[0]
        for i, eid in enumerate(self.env_id_list):
            assert env_id == eid, "policy {} and policy 0 is not trained in the same environment".format(i)
        return env_id

    def run(self):
        self.__run_data()
        self.__save_mp4_as_gif()
        self.draw()

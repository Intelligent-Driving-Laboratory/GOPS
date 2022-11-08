import argparse
import datetime
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import pandas as pd
from gym import wrappers
from copy import copy

from gops.create_pkg.create_env import create_env
from gops.utils.plot_evaluation import cm2inch
from gops.utils.common_utils import get_args_from_json, mp4togif
from gops.sys_simulator.sys_opt_controller import NNController

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


class PolicyRunner:
    def __init__(self, log_policy_dir_list, trained_policy_iteration_list, save_render=False, plot_range=None,
                 is_init_info=False, init_info=None, legend_list=None, use_opt=False, constrained_env=False,
                 is_tracking=False, use_dist=False, dt=None, obs_noise_type=None, obs_noise_data=None, action_noise_type=None, action_noise_data=None) -> None:
        self.log_policy_dir_list = log_policy_dir_list
        self.trained_policy_iteration_list = trained_policy_iteration_list
        self.save_render = save_render
        self.args = None
        self.plot_range = plot_range
        if is_init_info:
            self.init_info = init_info
        else:
            self.init_info = {}
        self.legend_list = legend_list
        self.use_opt = use_opt
        self.constrained_env = constrained_env
        self.use_dist = use_dist
        self.is_tracking = is_tracking
        self.dt = dt
        self.policy_num = len(self.log_policy_dir_list)
        if self.policy_num != len(self.trained_policy_iteration_list):
            raise RuntimeError("The lenth of policy number is not equal to the number of policy iteration")
        self.obs_noise_type = obs_noise_type
        self.obs_noise_data = obs_noise_data
        self.action_noise_type = action_noise_type
        self.action_noise_data = action_noise_data

        # data for plot

        #####################################################
        self.args_list = []
        self.eval_list = []
        self.env_id_list = []
        self.algorithm_list = []
        self.tracking_list = []

        self.__load_all_args()
        self.env_id = self.get_n_verify_env_id()

        if self.use_opt:
            self.legend_list.append('OPT')
            self.error_dict = {}
            self.opt_controller = NNController(self.args_list[0], self.log_policy_dir_list[0]) # TODO: replace with MPCController

        # save path
        path = os.path.join(os.path.dirname(__file__), "..", "..", "policy_result")
        path = os.path.abspath(path)

        algs_name = ""
        for item in self.algorithm_list:
            algs_name = algs_name + item + "-"
        self.save_path = os.path.join(path, algs_name + self.env_id, datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
        os.makedirs(self.save_path, exist_ok=True)

    def run_an_episode(self, env, controller, init_info, is_opt, render=True):
        state_list = []
        action_list = []
        reward_list = []
        constrain_list = []
        obs_list = []
        step = 0
        step_list = []
        info_list = [init_info]
        obs,_ = env.reset(**init_info)
        state = env.state
        print('The initial state is:')
        print(self.__convert_format(state))
        # plot tracking
        state_with_ref_error = {}
        done = False
        info = {"TimeLimit.truncated": False}
        while not (done or info["TimeLimit.truncated"]):
            state_list.append(state)
            obs_list.append(obs)
            if is_opt:
                if env.has_optimal_controller:
                    action = env.control_policy(state)
                else:
                    action = self.opt_controller(obs)
            else:
                action = self.compute_action(obs, controller)
                action = self.__action_noise(action)
            if self.use_dist:
                action = np.hstack((action, env.dist_func(step * env.tau)))
            next_obs, reward, done, info = env.step(action)

            action_list.append(action)
            step_list.append(step)
            reward_list.append(reward)
            info_list.append(info)

            obs = next_obs
            state = env.state
            step = step + 1
            # print("step:", step)

            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            # Draw environment animation
            if render:
                env.render()

            if self.constrained_env:
                constrain_list.append(info["constraint"])
            if self.is_tracking:
                state_num = len(info["ref"])
                self.ref_state_num = sum(x is not None for x in info["ref"])
                if step == 1:
                    for i in range(state_num):
                        if info["ref"][i] is not None:
                            state_with_ref_error["state-{}".format(i)] = []
                            state_with_ref_error["ref-{}".format(i)] = []
                            state_with_ref_error["state-{}-error".format(i)] = []

                for i in range(state_num):
                    if info["ref"][i] is not None:
                        state_with_ref_error["state-{}".format(i)].append(info["state"][i])
                        state_with_ref_error["ref-{}".format(i)].append(info["ref"][i])
                        state_with_ref_error["state-{}-error".format(i)].append(info["ref"][i] - info["state"][i])

        eval_dict = {
            "reward_list": reward_list,
            "action_list": action_list,
            "state_list": state_list,
            "step_list": step_list,
            "obs_list": obs_list,
            "info_list": info_list
        }
        if self.constrained_env:
            eval_dict.update({
                "constrain_list": constrain_list,
            })

        if self.is_tracking:
            tracking_dict = state_with_ref_error
        else:
            tracking_dict = {}

        return eval_dict, tracking_dict

    def compute_action(self, obs, networks):
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = networks.policy(batch_obs)
        action_distribution = networks.create_action_distributions(logits)
        action = action_distribution.mode()
        action = action.detach().numpy()[0]
        return action

    def draw(self):
        fig_size = (
            default_cfg["fig_size"],
            default_cfg["fig_size"],
        )
        action_dim = self.eval_list[0]["action_list"][0].shape[0]
        state_dim = self.eval_list[0]["state_list"][0].shape[0]
        if self.constrained_env:
            constrain_dim = self.eval_list[0]["constrain_list"][0].shape[0]
        policy_num = len(self.algorithm_list)
        if self.use_opt:
            policy_num += 1
            self.algorithm_list.append("OPT")

        # Create initial list
        reward_list = []
        action_list = []
        state_list = []
        step_list = []
        state_ref_error_list = []
        constrain_list = []
        # Put data into list
        for i in range(policy_num):
            reward_list.append(np.array(self.eval_list[i]["reward_list"]))
            action_list.append(np.array(self.eval_list[i]["action_list"]))
            state_list.append(np.array(self.eval_list[i]["state_list"]))
            step_list.append(np.array(self.eval_list[i]["step_list"]))
            if self.constrained_env:
                constrain_list.append(self.eval_list[i]["constrain_list"])
            if self.is_tracking:
                state_ref_error_list.append(self.tracking_list[i])

        if self.plot_range is None:
            pass
        elif len(self.plot_range) == 2:

            for i in range(policy_num):
                start_range = self.plot_range[0]
                end_range = min(self.plot_range[1], reward_list[i].shape[0])

                reward_list[i] = reward_list[i][start_range: end_range]
                action_list[i] = action_list[i][start_range: end_range]
                state_list[i] = state_list[i][start_range: end_range]
                step_list[i] = step_list[i][start_range: end_range]
                if self.constrained_env:
                    constrain_list[i] = constrain_list[i][start_range: end_range]
                if self.is_tracking:
                    for key, value in self.tracking_list[i].items():
                        self.tracking_list[i][key] = value[start_range: end_range]
        else:
            raise NotImplementedError("The setting of plot range is wrong")

        # Convert List to Array
        reward_array = np.array(reward_list)
        action_array = np.array(action_list)
        state_array = np.array(state_list)
        step_array = np.array(step_list)
        state_ref_error_array = np.array(state_ref_error_list)
        x_label = "Time Step"
        if self.dt is not None:
            step_array = step_array * self.dt
            x_label = "Time(s)"

        if self.constrained_env:
            constrain_array = np.array(constrain_list)

        # Plot reward
        path_reward_fmt = os.path.join(self.save_path, "Reward.{}".format(default_cfg["img_fmt"]))
        fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

        # save reward data to csv
        reward_data = pd.DataFrame(data=reward_array)
        reward_data.to_csv('{}\\Reward.csv'.format(self.save_path), encoding='gbk')

        for i in range(policy_num):
            legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
            sns.lineplot(x=step_array[i], y=reward_array[i], label="{}".format(legend))
        plt.tick_params(labelsize=default_cfg["tick_size"])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
        plt.xlabel(x_label, default_cfg["label_font"])
        plt.ylabel("Reward", default_cfg["label_font"])
        plt.legend(loc="best", prop=default_cfg["legend_font"])
        fig.tight_layout(pad=default_cfg["pad"])
        plt.savefig(path_reward_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

        # plot action
        for j in range(action_dim):
            path_action_fmt = os.path.join(self.save_path, "Action-{}.{}".format(j+1, default_cfg["img_fmt"]))
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save action data to csv
            action_data = pd.DataFrame(data=action_array[:, :, j])
            action_data.to_csv('{}\\Action-{}.csv'.format(self.save_path, j+1), encoding='gbk')

            for i in range(policy_num):
                legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                sns.lineplot(x=step_array[i], y=action_array[i, :, j],
                             label="{}".format(legend))
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Action-{}".format(j+1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_action_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

        # plot state
        for j in range(state_dim):
            path_state_fmt = os.path.join(self.save_path, "State-{}.{}".format(j+1, default_cfg["img_fmt"]))
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save state data to csv
            state_data = pd.DataFrame(data=state_array[:, :, j])
            state_data.to_csv('{}\\State-{}.csv'.format(self.save_path, j+1), encoding='gbk')

            for i in range(policy_num):
                legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                sns.lineplot(x=step_array[i], y=state_array[i, :, j],
                             label="{}".format(legend))
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("State-{}".format(j+1), default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

        # plot tracking
        if self.is_tracking:
            for j in range(self.ref_state_num):

                # plot state and ref
                path_tracking_state_fmt = os.path.join(self.save_path, "State-{}.{}".format(j + 1, default_cfg["img_fmt"]))
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
                # save tracking state data to csv
                tracking_state_data = []
                for i in range(policy_num):
                    legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                    sns.lineplot(x=step_array[i], y=state_ref_error_array[i]["state-{}".format(j)], label="{}".format(legend))
                    tracking_state_data.append(state_ref_error_array[i]["state-{}".format(j)])
                sns.lineplot(x=step_array[0], y=state_ref_error_array[0]["ref-{}".format(j)], label="ref")
                tracking_state_data.append(state_ref_error_array[0]["ref-{}".format(j)])
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{}".format(j + 1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(path_tracking_state_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

                tracking_state_data = pd.DataFrame(data=np.array(tracking_state_data))
                tracking_state_data.to_csv('{}\\State-{}.csv'.format(self.save_path, j + 1), encoding='gbk')

                # plot state-ref error
                path_tracking_error_fmt = os.path.join(self.save_path, "Ref - State-{}.{}".format(j+1, default_cfg["img_fmt"]))
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])
                # save tracking error data to csv
                tracking_error_data = []
                for i in range(policy_num):
                    legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                    sns.lineplot(x=step_array[i], y=state_ref_error_array[i]["state-{}-error".format(j)], label="{}".format(legend))
                    tracking_error_data.append(state_ref_error_array[i]["state-{}-error".format(j)])
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Ref $-$ State-{}".format(j+1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(path_tracking_error_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

                tracking_error_data = pd.DataFrame(data=np.array(tracking_error_data))
                tracking_error_data.to_csv('{}\\Ref - State-{}.csv'.format(self.save_path, j+1), encoding='gbk')

        # plot constraint value
        if self.constrained_env:
            for j in range(constrain_dim):
                path_constraint_fmt = os.path.join(self.save_path, "Constrain-{}.{}".format(j+1, default_cfg["img_fmt"]))
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

                # save reward data to csv
                constrain_data = pd.DataFrame(data=constrain_array[:, :, j])
                constrain_data.to_csv('{}\\Constrain-{}.csv'.format(self.save_path, j+1), encoding='gbk')

                for i in range(policy_num):
                    legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                    sns.lineplot(x=step_array[i], y=constrain_array[i, :, j], label="{}".format(legend))
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Constrain-{}".format(j+1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(path_constraint_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

        # plot error with opt
        if self.use_opt:
            # reward error
            path_reward_error_fmt = os.path.join(self.save_path, "Reward error.{}".format(default_cfg["img_fmt"]))
            fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

            # save reward error data to csv
            reward_error_array = reward_array[:-1] - reward_array[-1]
            reward_error_data = pd.DataFrame(data=reward_error_array)
            reward_error_data.to_csv('{}\\Reward error.csv'.format(self.save_path), encoding='gbk')

            for i in range(policy_num - 1):
                legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                sns.lineplot(x=step_array[i], y=reward_error_array[i], label="{}".format(legend))
            plt.tick_params(labelsize=default_cfg["tick_size"])
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
            plt.xlabel(x_label, default_cfg["label_font"])
            plt.ylabel("Reward error", default_cfg["label_font"])
            plt.legend(loc="best", prop=default_cfg["legend_font"])
            fig.tight_layout(pad=default_cfg["pad"])
            plt.savefig(path_reward_error_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

            # action error
            for j in range(action_dim):
                path_action_error_fmt = os.path.join(self.save_path, "Action-{} error.{}".format(j+1, default_cfg["img_fmt"]))
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

                action_error_array = np.zeros_like(action_array[:-1])

                for i in range(policy_num - 1):
                    action_error_array[i] = action_array[i] - action_array[-1]
                    legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                    sns.lineplot(x=step_array[i], y=action_error_array[i, :, j],
                                 label="{}".format(legend))
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("Action-{} error".format(j+1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(path_action_error_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

                # save action error data to csv
                action_error_data = pd.DataFrame(data=action_error_array[:, :, j])
                action_error_data.to_csv('{}\\Action-{} error.csv'.format(self.save_path, j+1), encoding='gbk')

            # state error
            for j in range(state_dim):
                path_state_error_fmt = os.path.join(self.save_path, "State-{} error.{}".format(j+1, default_cfg["img_fmt"]))
                fig, ax = plt.subplots(figsize=cm2inch(*fig_size), dpi=default_cfg["dpi"])

                state_error_array = np.zeros_like(state_array[:-1])

                for i in range(policy_num - 1):
                    state_error_array[i] = state_array[i] - state_array[-1]
                    legend = self.legend_list[i] if len(self.legend_list) == policy_num else self.algorithm_list[i]
                    sns.lineplot(x=step_array[i], y=state_error_array[i, :, j],
                                 label="{}".format(legend))
                plt.tick_params(labelsize=default_cfg["tick_size"])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontname(default_cfg["tick_label_font"]) for label in labels]
                plt.xlabel(x_label, default_cfg["label_font"])
                plt.ylabel("State-{} error".format(j+1), default_cfg["label_font"])
                plt.legend(loc="best", prop=default_cfg["legend_font"])
                fig.tight_layout(pad=default_cfg["pad"])
                plt.savefig(path_state_error_fmt, format=default_cfg["img_fmt"], bbox_inches="tight")

                # save state data to csv
                state_error_data = pd.DataFrame(data=state_error_array[:, :, j])
                state_error_data.to_csv('{}\\State-{} error.csv'.format(self.save_path, j+1), encoding='gbk')

            # compute relative error with opt
            error_result = {}
            # action error
            for i in range(policy_num - 1):
                legend = self.legend_list[i] if len(self.legend_list) == policy_num else "Policy-{}".format(i+1)
                error_result.update({legend: {}})
                # action error
                for j in range(action_dim):
                    action_error = {}
                    error_list = []
                    for q in range(len(action_array[0])):
                        error = np.abs(action_array[i, q, j] - action_array[-1, q, j]) \
                                / (np.max(action_array[-1, :, j]) - np.min(action_array[-1, :, j]))
                        error_list.append(error)
                    action_error["Max_error"] = '{:.2f}%'.format(max(error_list)*100)
                    action_error["Mean_error"] = '{:.2f}%'.format(sum(error_list) / len(error_list)*100)
                    error_result[legend].update({"Action-{}".format(j + 1): action_error})
                # state error
                for o in range(state_dim):
                    state_error = {}
                    error_list = []
                    for q in range(len(state_array[0])):
                        error = np.abs(state_array[i, q, o] - state_array[-1, q, o]) \
                                / (np.max(state_array[-1, :, o]) - np.min(state_array[-1, :, o]))
                        error_list.append(error)
                    state_error["Max_error"] = '{:.2f}%'.format(max(error_list)*100)
                    state_error["Mean_error"] = '{:.2f}%'.format(sum(error_list) / len(error_list)*100)
                    error_result[legend].update({"State-{}".format(o + 1): state_error})

            writer = pd.ExcelWriter('{}\\Error-result.xlsx'.format(self.save_path))
            for i in range(self.policy_num):
                legend = self.legend_list[i] if len(self.legend_list) == policy_num else "Policy-{}".format(i + 1)
                policy_result = pd.DataFrame(data=error_result[legend])
                policy_result.to_excel(writer, legend)
            writer.save()
            error_result_data = pd.DataFrame(data=error_result)
            # error_result_data.to_csv('{}\\Error-result.csv'.format(self.save_path), encoding='gbk')
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            for key, value in error_result_data.items():
                print("===========================================================")
                print('Policy {}'.format(key))
                for key, value in value.items():
                    print(key, value)




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

    def __load_env(self,use_opt =False):
        if use_opt:
            env = create_env( **self.args)
        else:
            env_args = {
                **self.args,
                "obs_noise_type": self.obs_noise_type,
                "obs_noise_data": self.obs_noise_data,
                "action_noise_type": self.action_noise_type,
                "action_noise_data": self.action_noise_data,
            }
            env = create_env(**env_args)
        if self.save_render:
            video_path = os.path.join(self.save_path, "videos")
            env = wrappers.RecordVideo(env, video_path,
                                       name_prefix="{}_video".format(self.args["algorithm"]))
        self.args["action_high_limit"] = env.action_space.high
        self.args["action_low_limit"] = env.action_space.low
        # self.args["has_controller"] = hasattr(env, 'has_optimal_controller') & env.has_optimal_controller
        return env

    def __load_policy(self, log_policy_dir, trained_policy_iteration):
        # Create policy
        alg_name = self.args["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        networks = ApproxContainer(**self.args)
        # print("Create {}-policy successfully!".format(alg_name))

        # Load trained policy
        log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(trained_policy_iteration)
        networks.load_state_dict(torch.load(log_path))
        # print("Load {}-policy successfully!".format(alg_name))
        return networks

    def __convert_format(self,origin_data_list:list):
        data_list = copy(origin_data_list)
        for i in range(len(origin_data_list)):
            if isinstance(origin_data_list[i],list) or isinstance(origin_data_list[i],np.ndarray) :
                data_list[i]= self.__convert_format(origin_data_list[i])
            else:
                data_list[i] = '{:.2g}'.format(origin_data_list[i])
        return(data_list)





    def __run_data(self):
        for i in range(self.policy_num):
            log_policy_dir = self.log_policy_dir_list[i]
            trained_policy_iteration = self.trained_policy_iteration_list[i]

            self.args = self.args_list[i]
            env = self.__load_env()
            print('The environment for policy {}'.format(i+1))
            if hasattr(env, 'train_space') and hasattr(env, 'work_space'):
                print('The train space is')
                print(self.__convert_format(env.train_space))
                print('The work space is')
                print(self.__convert_format(env.work_space))
            networks = self.__load_policy(log_policy_dir, trained_policy_iteration)

            # Run policy
            eval_dict, tracking_dict = self.run_an_episode(env, networks, self.init_info, is_opt=False, render=False)
            # mp4 to gif
            self.eval_list.append(eval_dict)
            self.tracking_list.append(tracking_dict)

        if self.use_opt:
            self.args = self.args_list[self.policy_num-1]
            env = self.__load_env(use_opt=True)
            print('The environment for opt')
            env.set_mode('test')
            controller = self.opt_controller
            eval_dict_opt, tracking_dict_opt = self.run_an_episode(env, controller, self.init_info, is_opt=True, render=False)
            self.eval_list.append(eval_dict_opt)
            if self.is_tracking:
                self.tracking_list.append(tracking_dict_opt)
            # opt_obs_list = eval_dict_opt["obs_list"]
            # opt_state_list = eval_dict_opt["state_list"]
            # opt_info_list = eval_dict_opt["info_list"]
            # self.obs_nums = len(opt_obs_list)
            # for i in range(self.policy_num):
            #     log_policy_dir = self.log_policy_dir_list[i]
            #     trained_policy_iteration = self.trained_policy_iteration_list[i]
            #     self.args = self.args_list[i]
            #     env = self.__load_env()
            #     networks = self.__load_policy(log_policy_dir, trained_policy_iteration)
            #     net_error_dict = self.__error_compute(env, opt_obs_list, opt_state_list,opt_info_list, networks, self.obs_nums, is_opt=False)
            #     LQ_error_dict = self.__error_compute(env, opt_obs_list, opt_state_list,opt_info_list, controller, self.obs_nums, is_opt=True)
            #     self.error_dict["policy_{}".format(i)] = net_error_dict
            #     self.error_dict["opt"] = LQ_error_dict


    def __get_init_info(self, env, init_state_nums):
        state_list = []
        obs_list = []
        for i in range(init_state_nums):
            obs = env.reset()
            obs_list.append(obs)
        return obs_list

    def __action_noise(self, action):
        if self.action_noise_type is None:
            return action
        elif self.action_noise_type == "normal":
            return action + np.random.normal(loc=self.action_noise_data[0], scale=self.action_noise_data[1])
        elif self.action_noise_type == "uniform":
            return action + np.random.uniform(low=self.action_noise_data[0], high=self.action_noise_data[1])

    def __error_compute(self, env, obs_list, state_list,info_list, controller, init_state_nums, is_opt):
        action_list = []
        next_state_list = []
        for i in range(init_state_nums):
            obs = obs_list[i]
            state = state_list[i]
            info_list[i].update({'init_state':state,'init_obs':obs})
            env.reset(**info_list[i])

            if is_opt:
                if env.has_optimal_controller:
                    action = env.control_policy(state)
                else:
                    action = controller(obs)
            else:
                action = self.compute_action(obs, controller)
                action = self.__action_noise(action)

            next_obs, reward, done, info = env.step(action)
            next_state = env.state
            action_list.append(action)
            next_state_list.append(next_state)
        action_array = np.array(action_list)
        next_state_array = np.array(next_state_list)
        error_dict = {"action": action_array, "next_state": next_state_array}

        return error_dict

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

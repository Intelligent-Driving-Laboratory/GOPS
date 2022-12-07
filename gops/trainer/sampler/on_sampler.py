#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Monte Carlo Sampler
#  Update Date: 2022-10-20, Yuheng Lei: Revise Codes
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes
#  Update Date: 2021-03-05, Wenxuan Wang: add action clip


import time

import numpy as np
import torch

from gops.create_pkg.create_env import create_env
from gops.utils.common_utils import set_seed
from gops.utils.explore_noise import GaussNoise, EpsilonGreedy
from gops.utils.tensorboard_setup import tb_tags


class OnSampler:
    def __init__(self, index=0, **kwargs):
        # initialize necessary hyperparameters
        self.env = create_env(**kwargs)
        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 200, self.env)
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        self.networks = ApproxContainer(**kwargs)
        self.noise_params = kwargs["noise_params"]
        self.sample_batch_size = kwargs["batch_size_per_sampler"]
        self.obs, self.info = self.env.reset()
        self.has_render = hasattr(self.env, "render")
        self.policy_func_name = kwargs["policy_func_name"]
        self.action_type = kwargs["action_type"]
        self.total_sample_number = 0
        self.obs_dim = kwargs["obsv_dim"]
        self.act_dim = kwargs["action_dim"]
        self.gamma = 0.99
        self.reward_scale = 1.0
        if isinstance(self.obs_dim, int):
            self.obs_dim = (self.obs_dim,)
        self.mb_obs = np.zeros(
            (self.sample_batch_size,) + self.obs_dim, dtype=np.float32
        )
        self.mb_act = np.zeros((self.sample_batch_size, self.act_dim), dtype=np.float32)
        self.mb_rew = np.zeros(self.sample_batch_size, dtype=np.float32)
        self.mb_done = np.zeros(self.sample_batch_size, dtype=np.bool_)
        self.mb_tlim = np.zeros(self.sample_batch_size, dtype=np.bool_)
        self.mb_logp = np.zeros(self.sample_batch_size, dtype=np.float32)
        self.need_value_flag = not (alg_name == "FHADP" or alg_name == "INFADP")
        if self.need_value_flag:
            self.gae_lambda = 0.95
            self.mb_val = np.zeros(self.sample_batch_size, dtype=np.float32)
            self.mb_adv = np.zeros(self.sample_batch_size, dtype=np.float32)
            self.mb_ret = np.zeros(self.sample_batch_size, dtype=np.float32)
        self.mb_info = {}
        self.info_keys = kwargs["additional_info"].keys()
        for k, v in kwargs["additional_info"].items():
            self.mb_info[k] = np.zeros(
                (self.sample_batch_size, *v["shape"]), dtype=v["dtype"]
            )
            self.mb_info["next_" + k] = np.zeros(
                (self.sample_batch_size, *v["shape"]), dtype=v["dtype"]
            )
        if self.noise_params is not None:
            if self.action_type == "continu":
                self.noise_processor = GaussNoise(**self.noise_params)
            elif self.action_type == "discret":
                self.noise_processor = EpsilonGreedy(**self.noise_params)

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def sample_with_replay_format(self):
        self.total_sample_number += self.sample_batch_size
        tb_info = dict()
        start_time = time.perf_counter()
        last_ptr, ptr = 0, 0
        for t in range(self.sample_batch_size):
            # output action using behavior policy
            obs_expand = torch.from_numpy(
                np.expand_dims(self.obs, axis=0).astype("float32")
            )
            logits = self.networks.policy(obs_expand)
            action_distribution = self.networks.create_action_distributions(logits)
            action, logp = action_distribution.sample()
            action = action.detach()[0].numpy()
            logp = logp.detach()[0].numpy()
            if self.noise_params is not None:
                action = self.noise_processor.sample(action)
            # ensure action is array
            action = np.array(action)
            if self.action_type == "continu":
                action_clip = action.clip(
                    self.env.action_space.low, self.env.action_space.high
                )
            else:
                action_clip = action
            # interact with environment
            next_obs, reward, self.done, next_info = self.env.step(action_clip)
            if "TimeLimit.truncated" not in next_info.keys():
                next_info["TimeLimit.truncated"] = False
            if next_info["TimeLimit.truncated"]:
                self.done = False
            reward *= self.reward_scale
            if self.need_value_flag:
                value = self.networks.value(obs_expand).detach().item()
                (
                    self.mb_obs[t],
                    self.mb_act[t],
                    self.mb_rew[t],
                    self.mb_done[t],
                    self.mb_tlim[t],
                    self.mb_logp[t],
                    self.mb_val[t],
                ) = (
                    self.obs.copy(),
                    action,
                    reward,
                    self.done,
                    next_info["TimeLimit.truncated"],
                    logp,
                    value,
                )
            else:
                (
                    self.mb_obs[t],
                    self.mb_act[t],
                    self.mb_rew[t],
                    self.mb_done[t],
                    self.mb_tlim[t],
                    self.mb_logp[t],
                ) = (
                    self.obs.copy(),
                    action,
                    reward,
                    self.done,
                    next_info["TimeLimit.truncated"],
                    logp,
                )
            for k in self.info_keys:
                self.mb_info[k][t] = self.info[k]
                self.mb_info["next_" + k][t] = next_info[k]
            self.obs = next_obs
            self.info = next_info
            if self.done or next_info["TimeLimit.truncated"]:
                self.obs, self.info = self.env.reset()
            # calculate value target (mb_ret) & gae (mb_adv)
            if (
                self.done
                or next_info["TimeLimit.truncated"]
                or t == self.sample_batch_size - 1
            ) and self.need_value_flag:
                last_obs_expand = torch.from_numpy(
                    np.expand_dims(next_obs, axis=0).astype("float32")
                )
                est_last_value = self.networks.value(
                    last_obs_expand
                ).detach().item() * (1 - self.done)
                ptr = t
                self._finish_trajs(est_last_value, last_ptr, ptr)
                last_ptr = t
        end_time = time.perf_counter()
        tb_info[tb_tags["sampler_time"]] = (end_time - start_time) * 1000
        # wrap collected data into replay format
        if self.need_value_flag:
            mb_data = {
                "obs": torch.from_numpy(self.mb_obs),
                "act": torch.from_numpy(self.mb_act),
                "rew": torch.from_numpy(self.mb_rew),
                "done": torch.from_numpy(self.mb_done),
                "logp": torch.from_numpy(self.mb_logp),
                "time_limited": torch.from_numpy(self.mb_tlim),
                "ret": torch.from_numpy(self.mb_ret),
                "adv": torch.from_numpy(self.mb_adv),
            }
        else:
            mb_data = {
                "obs": torch.from_numpy(self.mb_obs),
                "act": torch.from_numpy(self.mb_act),
                "rew": torch.from_numpy(self.mb_rew),
                "done": torch.from_numpy(self.mb_done),
                "logp": torch.from_numpy(self.mb_logp),
                "time_limited": torch.from_numpy(self.mb_tlim),
            }
        for k, v in self.mb_info.items():
            mb_data[k] = torch.from_numpy(v)
        return mb_data, tb_info

    def get_total_sample_number(self):
        return self.total_sample_number

    def _finish_trajs(self, est_last_val: float, last_ptr: int, ptr: int):
        # calculate value target (mb_ret) & gae (mb_adv) whenever episode is finished
        path_slice = slice(last_ptr, ptr + 1)
        value_preds_slice = np.append(self.mb_val[path_slice], est_last_val)
        rews_slice = self.mb_rew[path_slice]
        length = len(rews_slice)
        ret = np.zeros(length)
        adv = np.zeros(length)
        gae = 0.0
        for i in reversed(range(length)):
            delta = (
                rews_slice[i]
                + self.gamma * value_preds_slice[i + 1]
                - value_preds_slice[i]
            )
            gae = delta + self.gamma * self.gae_lambda * gae
            ret[i] = gae + value_preds_slice[i]
            adv[i] = gae
        self.mb_ret[path_slice] = ret
        self.mb_adv[path_slice] = adv

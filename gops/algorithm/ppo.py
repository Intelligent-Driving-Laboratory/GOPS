#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Proximal Policy Optimization (PPO) algorithm
#  Reference: Schulman J, Wolski F, Dhariwal P et al (2017) 
#             Proximal policy optimization algorithms. 
#             https://arxiv.org/abs/1707.06347.
#  Update: 2021-03-05, Yuxuan Jiang: create PPO algorithm


__all__ = ["ApproxContainer", "PPO"]


import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.gops_typing import DataDict
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags


class ApproxContainer(ApprBase):
    """Approximate function container for PPO.

    Contains one policy and one state value.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        policy_args = get_apprfunc_dict("policy", **kwargs)
        self.policy: nn.Module = create_apprfunc(**policy_args)
        value_args = get_apprfunc_dict("value", **kwargs)
        self.value: nn.Module = create_apprfunc(**value_args)

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class PPO(AlgorithmBase):
    """PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347

    :param max_iteration: Maximum iterations for learning rate schedule.
    :param num_repeat: Number of repeats (to reuse sample batch).
    :param num_mini_batch: Number of minibatches to divide sample batch.
    :param mini_batch_size: Minibatch size.
    :param sample_batch_size: Sample batch size.
    """

    def __init__(
        self,
        *,
        max_iteration: int,
        num_repeat: int,
        num_mini_batch: int,
        mini_batch_size: int,
        sample_batch_size: int,
        index=0,
        **kwargs
    ):
        super().__init__(index, **kwargs)
        self.max_iteration = max_iteration
        self.num_repeat = num_repeat
        self.num_mini_batch = num_mini_batch
        self.mini_batch_size = mini_batch_size
        self.sample_batch_size = sample_batch_size
        self.indices = np.arange(self.sample_batch_size)

        # Parameters for algorithm
        self.clip = 0.2
        self.clip_now = self.clip
        self.EPS = 1e-8
        self.gamma = 0.99
        self.reward_scale = 0.1
        self.loss_coefficient_kl = 0.2
        self.loss_coefficient_value = 1.0
        self.loss_coefficient_entropy = 0.0

        self.schedule_adam = "none"
        self.schedule_clip = "none"
        self.advantage_norm = True
        self.loss_value_clip = True
        self.value_clip = 10.0
        self.loss_value_norm = False

        self.networks = ApproxContainer(**kwargs)
        self.learning_rate = kwargs["learning_rate"]
        self.approximate_optimizer = Adam(
            self.networks.parameters(), lr=self.learning_rate
        )

    @property
    def adjustable_parameters(self):
        return (
            "gamma",
            "reward_scale",
            "clip",
            "loss_value_clip",
            "value_clip",
            "loss_value_norm",
            "advantage_norm",
            "loss_coefficient_kl",
            "loss_coefficient_value",
            "loss_coefficient_entropy",
            "schedule_adam",
            "schedule_clip",
        )

    def local_update(self, data: DataDict, iteration: int) -> dict:
        start_time = time.perf_counter()
        data["adv"] = (data["adv"] - data["adv"].mean()) / (
            data["adv"].std() + self.EPS
        )
        with torch.no_grad():
            data["val"] = self.networks.value(data["obs"])
            data["logits"] = self.networks.policy(data["obs"])

        for _ in range(self.num_repeat):
            np.random.shuffle(self.indices)

            for n in range(self.num_mini_batch):
                mb_start = self.mini_batch_size * n
                mb_end = self.mini_batch_size * (n + 1)
                mb_indices = self.indices[mb_start:mb_end]
                mb_sample = {k: v[mb_indices] for k, v in data.items()}
                (
                    loss_total,
                    loss_surrogate,
                    loss_value,
                    loss_entropy,
                    approximate_kl,
                    clip_fra,
                ) = self.__compute_loss(mb_sample, iteration)
                self.approximate_optimizer.zero_grad()
                loss_total.backward()
                self.approximate_optimizer.step()
                if self.schedule_adam == "linear":
                    decay_rate = 1 - (iteration / self.max_iteration)
                    assert decay_rate >= 0, "the decay_rate is less than 0!"
                    lr_now = self.learning_rate * decay_rate
                    # set learning rate
                    for g in self.approximate_optimizer.param_groups:
                        g["lr"] = lr_now

        end_time = time.perf_counter()

        tb_info = dict()
        tb_info[tb_tags["loss_actor"]] = loss_surrogate.item()
        tb_info[tb_tags["loss_critic"]] = loss_value.item()
        tb_info["PPO/KL_divergence-RL iter"] = approximate_kl.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000

        return tb_info

    def __compute_loss(self, data: DataDict, iteration: int):
        obs, act = data["obs"], data["act"]
        pro = data["logp"]
        returns, advantages, values = data["ret"], data["adv"], data["val"]
        logits = data["logits"]

        # name completion
        mb_observation = obs
        mb_action = act
        mb_old_log_pro = pro
        mb_old_logits = logits
        mb_old_act_dist = self.networks.create_action_distributions(mb_old_logits)
        mb_new_logits = self.networks.policy(mb_observation)
        mb_new_act_dist = self.networks.create_action_distributions(mb_new_logits)
        mb_new_log_pro = mb_new_act_dist.log_prob(mb_action)

        assert not advantages.requires_grad and not returns.requires_grad
        mb_return = returns.detach()
        mb_advantage = advantages.detach()
        mb_old_value = values
        mb_new_value = self.networks.value(mb_observation)

        # policy loss
        ratio = torch.exp(mb_new_log_pro - mb_old_log_pro)
        sur1 = ratio * mb_advantage
        sur2 = ratio.clamp(1 - self.clip_now, 1 + self.clip_now) * mb_advantage
        loss_surrogate = -torch.mean(torch.min(sur1, sur2))

        if self.loss_value_clip:
            # unclipped value
            value_losses1 = torch.pow(mb_new_value - mb_return, 2)
            # clipped value
            mb_new_value_clipped = mb_old_value + (mb_new_value - mb_old_value).clamp(
                -self.value_clip, self.value_clip
            )
            value_losses2 = torch.pow(mb_new_value_clipped - mb_return, 2)
            # value loss
            value_losses = torch.max(value_losses1, value_losses2)
        else:
            value_losses = torch.pow(mb_new_value - mb_return, 2)
        if self.loss_value_norm:
            mb_return_6std = 6 * mb_return.std()
            loss_value = torch.mean(value_losses) / mb_return_6std
        else:
            loss_value = torch.mean(value_losses)

        # entropy loss
        loss_entropy = torch.mean(mb_new_act_dist.entropy())
        loss_kl = torch.mean(mb_old_act_dist.kl_divergence(mb_new_act_dist))
        clip_fraction = torch.mean(
            torch.gt(torch.abs(ratio - 1.0), self.clip_now).float()
        )

        # total loss
        loss_total = (
            loss_surrogate
            + self.loss_coefficient_kl * loss_kl
            + self.loss_coefficient_value * loss_value
            - self.loss_coefficient_entropy * loss_entropy
        )

        if self.schedule_clip == "linear":
            decay_rate = 1 - (iteration / self.max_iteration)
            assert decay_rate >= 0, "decay_rate is less than 0!"
            self.clip_now = self.clip * decay_rate

        return (
            loss_total,
            loss_surrogate,
            loss_value,
            loss_entropy,
            loss_kl,
            clip_fraction,
        )

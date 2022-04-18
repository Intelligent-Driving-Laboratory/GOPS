#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Distributed Soft Actor Critic Algorithm (DSAC)
#  Update Date: 2021-03-05, Ziqing Gu: create DSAC algorithm
#  Update Date: 2021-03-05, Wenxuan Wang: debug DSAC algorithm


__all__ = ["ApproxContainer", "DSAC"]

import time
from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.tensorboard_tools import tb_tags
from gops.utils.utils import get_apprfunc_dict


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create q networks
        q_func_type = kwargs["value_func_type"]
        q_args = get_apprfunc_dict("q", q_func_type, **kwargs)
        self.q = create_apprfunc(**q_args)
        self.q_target = deepcopy(self.q)

        # create policy network
        policy_func_type = kwargs["policy_func_type"]
        policy_args = get_apprfunc_dict("policy", policy_func_type, **kwargs)
        self.policy = create_apprfunc(**policy_args)
        self.policy_target = deepcopy(self.policy)

        # set target network gradients
        for p in self.policy_target.parameters():
            p.requires_grad = False
        for p in self.q_target.parameters():
            p.requires_grad = False

        # create entropy coefficient
        self.log_alpha = nn.Parameter(torch.tensor(0, dtype=torch.float32))

        # create optimizers
        self.q_optimizer = Adam(self.q.parameters(), lr=kwargs["q_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )
        self.alpha_optimizer = Adam(
            [self.log_alpha], lr=kwargs["alpha_learning_rate"]
        )

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class DSAC(AlgorithmBase):
    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.gamma = kwargs["gamma"]
        self.tau = kwargs["tau"]
        self.reward_scale = kwargs["reward_scale"]
        self.target_entropy = -kwargs["action_dim"]
        self.auto_alpha = kwargs["auto_alpha"]
        self.TD_bound = kwargs["TD_bound"]
        self.bound = kwargs["bound"]
        self.delay_update = kwargs["delay_update"]
        self.alpha = 0.2

    def get_alpha(self, requires_grad=False):
        if self.auto_alpha:
            alpha = self.networks.log_alpha.exp()
            if requires_grad:
                return alpha
            else:
                return alpha.item()
        else:
            return self.alpha

    @property
    def adjustable_parameters(self):
        para_tuple = ("gamma", "tau", "reward_scale", "auto_alpha", "alpha",
                      "TD_bound", "bound", "delay_update")
        return para_tuple

    def __compute_gradient(self, data: dict, iteration):
        start_time = time.time()
        data["rew"] = data["rew"] * self.reward_scale

        obs = data["obs"]
        logits = self.networks.policy(obs)
        policy_mean = torch.tanh(logits[..., 0]).mean().item()
        policy_std = logits[..., 1].mean().item()

        act_dist = self.networks.create_action_distributions(logits)
        new_act, new_log_prob = act_dist.rsample()
        data.update({"new_act": new_act, "new_log_prob": new_log_prob})

        self.networks.q_optimizer.zero_grad()
        loss_q, q, std = self.__compute_loss_q(data)
        loss_q.backward()

        for p in self.networks.q.parameters():
            p.requires_grad = False

        self.networks.policy_optimizer.zero_grad()
        loss_policy, entropy = self.__compute_loss_policy(data)
        loss_policy.backward()

        for p in self.networks.q.parameters():
            p.requires_grad = True

        tb_info = {
            "DSAC/critic_avg_q-RL iter": q.item(),
            "DSAC/critic_avg_std-RL iter": std.item(),
            tb_tags["loss_actor"]: loss_policy.item(),
            "DSAC/policy_mean-RL iter": policy_mean,
            "DSAC/policy_std-RL iter": policy_std,
            "DSAC/entropy-RL iter": entropy.item(),
            "DSAC/alpha-RL iter": self.get_alpha(),
            tb_tags["alg_time"]: (time.time() - start_time) * 1000,
        }

        return tb_info

    def __q_evaluate(self, obs, act, qnet, min=False):
        StochaQ = qnet(obs, act)
        mean, log_std = StochaQ[..., 0], StochaQ[..., -1]
        std = log_std.exp()
        normal = Normal(torch.zeros(mean.shape), torch.ones(std.shape))
        if min:
            z = -torch.abs(normal.sample())
        else:
            z = normal.sample()
            z = torch.clamp(z, -2, 2)
        q_value = mean + torch.mul(z, std)  # + torch.mul(z, std)
        return mean, std, q_value

    def __compute_loss_q(self, data):
        obs, act, rew, obs2, done = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        logits_2 = self.networks.policy_target(obs2)
        act2_dist = self.networks.create_action_distributions(logits_2)
        act2, log_prob_act2 = act2_dist.rsample()

        q, q_std, q_sample = self.__q_evaluate(obs, act, self.networks.q, min=False)
        # _, _, q_next_sample = self._q_evaluate(obs2, act2, self.networks.q_target, min=False)
        _, _, q_next_sample = self.__q_evaluate(
            obs2, act2, self.networks.q_target, min=False
        )
        target_q, target_q_bound = self.__compute_target_q(
            rew,
            done,
            q.detach(),
            q_std.detach(),
            q_next_sample.detach(),
            log_prob_act2.detach(),
        )
        if self.bound:
            q_loss = torch.mean(
                torch.pow(q - target_q, 2) / (2 * torch.pow(q_std.detach(), 2))
                + torch.pow(q.detach() - target_q_bound, 2) / (2 * torch.pow(q_std, 2))
                + torch.log(q_std)
            )
        else:
            q_loss = -Normal(q, q_std).log_prob(target_q).mean()
        return q_loss, q.detach().mean(), q_std.detach().mean()

    def __compute_target_q(self, r, done, q, q_std, q_next, log_prob_a_next):
        target_q = r + (1 - done) * self.gamma * (
                q_next - self.get_alpha() * log_prob_a_next
        )
        difference = torch.clamp(target_q - q, -self.TD_bound, self.TD_bound)
        target_q_bound = q + difference
        return target_q.detach(), target_q_bound.detach()

    def __compute_loss_policy(self, data):
        obs, new_act, new_log_prob = data["obs"], data["new_act"], data["new_log_prob"]
        q, _, _ = self.__q_evaluate(obs, new_act, self.networks.q, min=False)
        loss_policy = (self.alpha * new_log_prob - q).mean()
        entropy = -new_log_prob.detach().mean()
        return loss_policy, entropy

    def __compute_loss_alpha(self, data):
        new_log_prob = data["new_log_prob"]
        loss_alpha = (
                -self.get_alpha(requires_grad=True) * (new_log_prob.detach() + self.target_entropy).mean()
        )
        return loss_alpha

    def __update(self, iteration):
        self.networks.q_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()

            if self.auto_alpha:
                self.networks.alpha_optimizer.step()

        with torch.no_grad():
            polyak = 1 - self.tau
            for p, p_targ in zip(self.networks.q.parameters(), self.networks.q_target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(
                    self.networks.policy.parameters(), self.networks.policy_target.parameters()
            ):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def local_update(self, data: dict, iteration: int):
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            "q_grad": [p._grad for p in self.networks.q.parameters()],
            "policy_grad": [p._grad for p in self.networks.policy.parameters()],
            "iteration": iteration,
        }

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q_grad = update_info["q_grad"]
        policy_grad = update_info["policy_grad"]

        for p, grad in zip(self.networks.q.parameters(), q_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad

        self.__update(iteration)


if __name__ == "__main__":
    print("this is dsac algorithm!")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

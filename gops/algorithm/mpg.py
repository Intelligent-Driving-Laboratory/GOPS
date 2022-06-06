#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Mixed Policy Gradient Algorithm (MPG)
#  Update Date: 2022-06-05, Ziqing Gu: create MPG algorithm


__all__ = ["ApproxContainer", "MPG"]

import time
from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
from torch.optim import Adam

from gops.algorithm.base import AlgorithmBase, ApprBase
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.tensorboard_tools import tb_tags
from gops.utils.utils import get_apprfunc_dict


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # create value network
        value_func_type = kwargs["value_func_type"]
        pge_method = kwargs["pge_method"]

        q_args = get_apprfunc_dict("value", value_func_type, **kwargs)
        self.q1 = create_apprfunc(**q_args)
        self.q2 = create_apprfunc(**q_args)
        if pge_method == 'mixed_state':
            self.q1_model = deepcopy(self.q1)
            self.q2_model = deepcopy(self.q2)

        # create policy network
        policy_func_type = kwargs["policy_func_type"]
        policy_args = get_apprfunc_dict("policy", policy_func_type, **kwargs)
        self.policy = create_apprfunc(**policy_args)
        self.policy4rollout = create_apprfunc(**policy_args)
        self.policy4rollout = deepcopy(self.policy)
        for p in self.policy4rollout.parameters():
            p.requires_grad = False

        #  create target networks
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        if pge_method == 'mixed_state':
            self.q1_model_target = deepcopy(self.q1_model)
            self.q2_model_target = deepcopy(self.q2_model)
        self.policy_target = deepcopy(self.policy)

        # set target network gradients
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False
        if pge_method == 'mixed_state':
            for p in self.q1_model_target.parameters():
                p.requires_grad = False
            for p in self.q2_model_target.parameters():
                p.requires_grad = False

        # set optimizers
        self.q1_optimizer = Adam(self.q1.parameters(), lr=kwargs["value_learning_rate"])
        self.q2_optimizer = Adam(self.q2.parameters(), lr=kwargs["value_learning_rate"])
        if pge_method == 'mixed_state':
            self.q1_model_optimizer = Adam(self.q1_model.parameters(), lr=kwargs["value_learning_rate"])
            self.q2_model_optimizer = Adam(self.q2_model.parameters(), lr=kwargs["value_learning_rate"])
        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)


class MPG(AlgorithmBase):
    def __init__(self, index=0, **kwargs):
        super(MPG, self).__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.pge_method = kwargs["pge_method"]
        if self.pge_method == 'mixed_weight':
            self.terminal_iter = kwargs.get("terminal_iter", 10000)
            self.eta = kwargs.get("eta", 0.1)
        elif self.pge_method == 'mixed_state':
            self.kappa = kwargs.get("kappa", 0.5)
        self.gamma = kwargs["gamma"]
        self.tau = kwargs["tau"]

        self.reward_scale = kwargs["reward_scale"]
        self.delay_update = kwargs["delay_update"]
        self.forward_step = kwargs["forward_step"]

    @property
    def adjustable_parameters(self):
        para_tuple = ("gamma", "tau", "delay_update", "reward_scale",
                      "terminal_iter", "eta")
        return para_tuple

    def __compute_gradient(self, data: dict, iteration):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"] * self.reward_scale,
            data["obs2"],
            data["done"],
        )
        self.networks.q1_optimizer.zero_grad()
        self.networks.q2_optimizer.zero_grad()
        self.networks.policy_optimizer.zero_grad()
        if self.pge_method == "mixed_state":
            self.networks.q1_model_optimizer.zero_grad()
            self.networks.q2_model_optimizer.zero_grad()

        start_time = time.time()
        q_info, backup_info = self.__compute_loss_q(o, a, r, o2, d)
        loss_q = q_info['MPG/loss_q-RL iter']
        loss_q.backward()
        if self.pge_method == "mixed_state":
            loss_q_model = q_info['MPG/loss_q_model-RL iter']
            loss_q_model.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = False
        for p in self.networks.q2.parameters():
            p.requires_grad = False
        if self.pge_method == "mixed_state":
            for p in self.networks.q1_model.parameters():
                p.requires_grad = False
            for p in self.networks.q2_model.parameters():
                p.requires_grad = False

        loss_pi, pi_tb_info = self.__compute_loss_pi(data, iteration, backup_info)
        loss_pi.backward()

        for p in self.networks.q1.parameters():
            p.requires_grad = True
        for p in self.networks.q2.parameters():
            p.requires_grad = True
        if self.pge_method == "mixed_state":
            for p in self.networks.q1_model.parameters():
                p.requires_grad = True
            for p in self.networks.q2_model.parameters():
                p.requires_grad = True

        end_time = time.time()
        tb_info = {tb_tags["alg_time"]: (end_time - start_time) * 1000}
        q_tb_info = {k: v.item() for k, v in q_info.items()}
        tb_info.update(q_tb_info)
        tb_info.update(pi_tb_info)
        return tb_info

    def __compute_value_backup(self, o, a, r, o2, d):
        with torch.no_grad():
            pi_targ = self.networks.policy_target(o2)
            # Target Q-values
            q1_pi_targ = self.networks.q1_target(o2, pi_targ)
            q2_pi_targ = self.networks.q2_target(o2, pi_targ)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * q_pi_targ
        return backup

    def __compute_value_backup_model(self, o, a, r, o2, d):
        with torch.no_grad():
            pi_targ = self.networks.policy_target(o2)
            # Target Q-values of model
            q1_model_pi_targ = self.networks.q1_model_target(o2, pi_targ)
            q2_model_pi_targ = self.networks.q2_model_target(o2, pi_targ)
            q_model_pi_targ = torch.min(q1_model_pi_targ, q2_model_pi_targ)
            backup_model = r + self.gamma * (1 - d) * q_model_pi_targ
        return backup_model

    def __compute_loss_q(self, o, a, r, o2, d):
        q1 = self.networks.q1(o, a)
        q2 = self.networks.q2(o, a)

        # Bellman backup for Q functions
        backup_data = self.__compute_value_backup(o, a, r, o2, d)
        backup_info = {'backup_data': backup_data}

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup_data) ** 2).mean()
        loss_q2 = ((q2 - backup_data) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        q_info = {"MPG/loss_q1-RL iter": loss_q1,
                  "MPG/loss_q2-RL iter": loss_q2,
                  "MPG/loss_q-RL iter": loss_q,
                  "MPG/q1_mean-RL iter": q1.mean(),
                  "MPG/q2_mean-RL iter": q2.mean(),
                  }

        if self.pge_method == "mixed_state":
            q1_model = self.networks.q1_model(o, a)
            q2_model = self.networks.q2_model(o, a)

            # Bellman backup for Q functions
            backup_model = self.__compute_value_backup_model(o, a, r, o2, d)
            backup_info.update({'backup_model': backup_model})
            # MSE loss against Bellman backup
            loss_q1_model = ((q1_model - backup_model) ** 2).mean()
            loss_q2_model = ((q2_model - backup_model) ** 2).mean()
            loss_q_model = loss_q1_model + loss_q2_model
            q_info.update({"MPG/loss_q1_model-RL iter": loss_q1_model,
                           "MPG/loss_q2_model-RL iter": loss_q2_model,
                           "MPG/loss_q_model-RL iter": loss_q_model,
                           "MPG/q1_model_mean-RL iter": q1_model.mean(),
                           "MPG/q2_model_mean-RL iter": q2_model.mean(),
                           })
        return q_info, backup_info

    def __compute_weights(self, iteration):
        start = 1. - self.eta
        slope = 2. * self.eta / self.terminal_iter
        lam = start + slope * iteration
        lam = np.clip(lam, 0, 1.5)
        if lam < 1.:
            biases = np.array([np.power(lam, i) for i in [0, self.forward_step]])
        else:
            max_index = self.forward_step
            biases = np.array([np.power(2 - lam, max_index - i) for i in [0, self.forward_step]])
        bias_inverses = 1. / (biases + 1e-8)
        ws = torch.softmax(torch.tensor(bias_inverses), dim=0)
        return ws

    def __compute_loss_pi(self, data, iteration, backup_info):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        data_return = self.networks.q1(o, self.networks.policy(o))
        model_return = torch.zeros(1)
        for step in range(self.forward_step):
            if step == 0:
                a = self.networks.policy(o)
                o2, r, _, _ = self.envmodel.forward(o, a, torch.tensor(0))
                model_return = self.reward_scale * r
            else:
                o = o2
                a = self.networks.policy4rollout(o)
                o2, r, _, _ = self.envmodel.forward(o, a, torch.tensor(0))
                model_return += self.reward_scale * self.gamma ** step * r
        model_return += self.gamma ** self.forward_step * self.networks.q1_target(o2, self.networks.policy(o2))
        if self.pge_method == "mixed_weight":
            with torch.no_grad():
                ws = self.__compute_weights(iteration)
            data_w, model_w = ws[0], ws[1]
            data_loss = -data_return.mean()
            model_loss = -model_return.mean()
            loss = data_w * data_loss + model_w * model_loss
            pi_tb_info = {
                "MPG/data_w-RL iter": data_w.item(),
                "MPG/model_w-RL iter": model_w.item(),
                "MPG/data_loss-RL iter": data_loss.item(),
                "MPG/model_loss-RL iter": model_loss.item(),
                "MPG/loss_pi-RL iter": loss.item()}
        else:
            assert self.pge_method == "mixed_state", "the pge_method entry should be mixed_state or mixed_weight"
            backup_data, backup_model = backup_info['backup_data'], backup_info['backup_model']
            with torch.no_grad():
                model_condi = torch.abs_(backup_data - backup_model) < self.kappa * backup_data.std()
            loss = torch.where(model_condi, -model_return, -data_return).mean()
            model_ratio = model_condi.float().mean()
            pi_tb_info = {
                "MPG/model_ratio-RL iter": model_ratio.item(),
                "MPG/data_loss-RL iter": -data_return.mean().item(),
                "MPG/model_loss-RL iter": -model_return.mean().item(),
                "MPG/loss_pi-RL iter": loss.item()}
        return loss, pi_tb_info

    def __update(self, iteration):
        self.networks.q1_optimizer.step()
        self.networks.q2_optimizer.step()
        if self.pge_method == "mixed_state":
            self.networks.q1_model_optimizer.step()
            self.networks.q2_model_optimizer.step()

        if iteration % self.delay_update == 0:
            self.networks.policy_optimizer.step()
        self.networks.policy4rollout = deepcopy(self.networks.policy)
        for p in self.networks.policy4rollout.parameters():
            p.requires_grad = False
        with torch.no_grad():
            polyak = 1 - self.tau
            for p, p_targ in zip(self.networks.q1.parameters(), self.networks.q1_target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(self.networks.q2.parameters(), self.networks.q2_target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            for p, p_targ in zip(self.networks.policy.parameters(), self.networks.policy_target.parameters()):
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
            if self.pge_method == "mixed_state":
                for p, p_targ in zip(self.networks.q1_model.parameters(), self.networks.q1_model_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
                for p, p_targ in zip(self.networks.q2_model.parameters(), self.networks.q2_model_target.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def local_update(self, data: dict, iteration: int):
        tb_info = self.__compute_gradient(data, iteration)
        self.__update(iteration)
        return tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        tb_info = self.__compute_gradient(data, iteration)

        update_info = {
            "q1_grad": [p._grad for p in self.networks.q1.parameters()],
            "q2_grad": [p._grad for p in self.networks.q2.parameters()],
            "policy_grad": [p._grad for p in self.networks.policy.parameters()],
            "iteration": iteration,
        }
        if self.pge_method == "mixed_state":
            update_info.update(
                {"q1_model_grad": [p._grad for p in self.networks.q1_model.parameters()],
                 "q2_model_grad": [p._grad for p in self.networks.q2_model.parameters()]})

        return tb_info, update_info

    def remote_update(self, update_info: dict):
        iteration = update_info["iteration"]
        q1_grad = update_info["q1_grad"]
        q2_grad = update_info["q2_grad"]
        policy_grad = update_info["policy_grad"]

        for p, grad in zip(self.networks.q1.parameters(), q1_grad):
            p._grad = grad
        for p, grad in zip(self.networks.q2.parameters(), q2_grad):
            p._grad = grad
        for p, grad in zip(self.networks.policy.parameters(), policy_grad):
            p._grad = grad

        if self.pge_method == "mixed_state":
            q1_model_grad = update_info["q1_model_grad"]
            q2_model_grad = update_info["q2_model_grad"]
            for p, grad in zip(self.networks.q1_model.parameters(), q1_model_grad):
                p._grad = grad
            for p, grad in zip(self.networks.q2_model.parameters(), q2_model_grad):
                p._grad = grad
        self.__update(iteration)


if __name__ == "__main__":
    print("this is mpg algorithm!")

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Infinity ADP Algorithm (infADP)
#  Update: 2021-03-05, Wenxuan Wang: create infADP algorithm

__all__ = ["INFADP"]

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam
import time
import warnings

from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.utils import get_apprfunc_dict
from gops.utils.tensorboard_tools import tb_tags
from gops.utils.utils import set_seed

class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        value_func_type = kwargs["value_func_type"]
        policy_func_type = kwargs["policy_func_type"]

        if kwargs["cnn_shared"]:  # todo:设置默认false
            feature_args = get_apprfunc_dict("feature", value_func_type, **kwargs)
            kwargs["feature_net"] = create_apprfunc(**feature_args)

        v_args = get_apprfunc_dict("value", value_func_type, **kwargs)
        policy_args = get_apprfunc_dict("policy", policy_func_type, **kwargs)

        self.v = create_apprfunc(**v_args)
        self.policy = create_apprfunc(**policy_args)

        self.v_target = deepcopy(self.v)
        self.policy_target = deepcopy(self.policy)

        for p in self.v_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False

        self.policy_optimizer = Adam(
            self.policy.parameters(), lr=kwargs["policy_learning_rate"]
        )  #
        self.v_optimizer = Adam(self.v.parameters(), lr=kwargs["value_learning_rate"])

        self.net_dict = {"v": self.v, "policy": self.policy}
        self.target_net_dict = {"v": self.v_target, "policy": self.policy_target}
        self.optimizer_dict = {"v": self.v_optimizer, "policy": self.policy_optimizer}

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)

    def update(self, grad_info):
        tau = grad_info["tau"]
        grads_dict = grad_info["grads_dict"]
        for net_name, grads in grads_dict.items():
            for p, grad in zip(self.net_dict[net_name].parameters(), grads):
                p.grad = grad
            self.optimizer_dict[net_name].step()

        with torch.no_grad():
            for net_name in grads_dict.keys():
                for p, p_targ in zip(
                    self.net_dict[net_name].parameters(),
                    self.target_net_dict[net_name].parameters(),
                ):
                    p_targ.data.mul_(1 - tau)
                    p_targ.data.add_(tau * p.data)


class INFADP:
    def __init__(self, index=0, **kwargs):
        set_seed(kwargs["trainer"], kwargs["seed"], index + 300)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.envmodel = self.envmodel.cuda()
        self.gamma = 0.99
        self.tau = 0.005
        self.pev_step = 1
        self.pim_step = 1
        self.forward_step = 10
        self.reward_scale = 0.1
        self.tb_info = dict()

    def set_parameters(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
            else:
                warning_msg = "param '" + key + "'is not defined in algorithm!"
                warnings.warn(warning_msg)

    def get_parameters(self):
        params = dict()
        params["use_gpu"] = self.use_gpu
        params["gamma"] = self.gamma
        params["tau"] = self.tau
        params["pev_step"] = self.pev_step
        params["pim_step"] = self.pim_step
        params["reward_scale"] = self.reward_scale
        params["forward_step"] = self.forward_step
        return params

    def compute_gradient(self, data, iteration):
        grad_info = dict()
        grads_dict = dict()

        start_time = time.time()
        if self.use_gpu:
            self.networks = self.networks.cuda()
            for key, value in data.items():
                data[key] = value.cuda()

        if iteration % (self.pev_step + self.pim_step) < self.pev_step:
            self.networks.v.zero_grad()
            loss_v, v = self.compute_loss_v(data)
            loss_v.backward()
            v_grad = [p.grad for p in self.networks.v.parameters()]
            self.tb_info[tb_tags["loss_critic"]] = loss_v.item()
            self.tb_info[tb_tags["critic_avg_value"]] = v.item()
            grads_dict["v"] = v_grad
        else:
            self.networks.policy.zero_grad()
            loss_policy = self.compute_loss_policy(data)
            loss_policy.backward()
            policy_grad = [p.grad for p in self.networks.policy.parameters()]
            self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()
            grads_dict["policy"] = policy_grad

        if self.use_gpu:
            self.networks = self.networks.cpu()
            for key, value in data.items():
                data[key] = value.cpu()

        end_time = time.time()

        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

        grad_info["tau"] = self.tau
        grad_info["grads_dict"] = grads_dict
        return grad_info, self.tb_info

    def compute_loss_v(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )  # TODO  解耦字典
        v = self.networks.v(o)

        with torch.no_grad():
            for step in range(self.forward_step):
                if step == 0:
                    a = self.networks.policy(o)
                    o2, r, d, _ = self.envmodel.forward(o, a, d)
                    backup = self.reward_scale * r
                else:
                    o = o2
                    a = self.networks.policy(o)
                    o2, r, d, _ = self.envmodel.forward(o, a, d)
                    backup += self.reward_scale * self.gamma ** step * r

            backup += (
                (~d) * self.gamma ** self.forward_step * self.networks.v_target(o2)
            )
        loss_v = ((v - backup) ** 2).mean()
        return loss_v, torch.mean(v)

    def compute_loss_policy(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )  # TODO  解耦字典
        v_pi = torch.zeros(1)
        for p in self.networks.v.parameters():
            p.requires_grad = False
        for step in range(self.forward_step):
            if step == 0:
                a = self.networks.policy(o)
                o2, r, d, _ = self.envmodel.forward(o, a, d)
                v_pi = self.reward_scale * r
            else:
                o = o2
                a = self.networks.policy(o)
                o2, r, d, _ = self.envmodel.forward(o, a, d)
                v_pi += self.reward_scale * self.gamma ** step * r
        v_pi += (~d) * self.gamma ** self.forward_step * self.networks.v_target(o2)
        for p in self.networks.v.parameters():
            p.requires_grad = True
        return -v_pi.mean()

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)


if __name__ == "__main__":
    print("11111")

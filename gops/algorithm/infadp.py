#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Approximate Dynamic Program Algorithm for Infinity Horizon (INFADP)
#  Reference: Vamvoudakis K, Lewis F, Hudas G (2012) Multi-agent differential graphical
#             games: Online adaptive learning solution for synchronization with optimality.
#             Pergamon, Oxford
#  Update: 2021-03-05, Wenxuan Wang: create infADP algorithm
#  Update: 2022-12-04, Jiaxin Gao: supplementary comment information

__all__ = ["INFADP"]

from copy import deepcopy
from typing import Tuple

import torch
from torch.optim import Adam
import time

from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.common_utils import get_apprfunc_dict
from gops.utils.tensorboard_setup import tb_tags
from gops.algorithm.base import AlgorithmBase, ApprBase


class ApproxContainer(ApprBase):
    def __init__(self, **kwargs):
        """Approximate function container for INFADP."""
        """Contains two policy and two action values."""

        super().__init__(**kwargs)
        value_func_type = kwargs["value_func_type"]
        policy_func_type = kwargs["policy_func_type"]

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
        """create action distribution"""
        return self.policy.get_act_dist(logits)


class INFADP(AlgorithmBase):
    """Approximate Dynamic Program Algorithm for Infinity Horizon
    Paper: https://pdf.sciencedirectassets.com/271426/1-s2.0-S0005109812X00086/1-s2.0-S0005109812002476/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjELv%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIGKabUjDfNnvJwC%2BF9rMRzaIYxIFr58rQLDd9TbLjcDBAiArEWmT%2FgZlB%2BK5tZfI54BqbSEwHSoQLPuqKFfOWqXzqirVBAjU%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMyanIgObQqXvQu4ypKqkEXMWSDU47sXj92wVxRHA%2BgKN%2BNetV93KWQlkNPbSzdVMf2v3cmS7q0UxOFH9EZvr7wJJWFxmSbHsNE8mFJEm0xVvT0tAMrOV08h3xKFYAHR6b7WH7mVGl7jWGQQooDPVgwZxx5Gs1NJaeInMP4nTuYKRWHDtGGI9eE3ji1YypCNPyEFB6iHXq32akl3yMiOGFib9SKEV%2BFRSazUD1E06c91q8SAegjT%2F8WWCSzhENO7TuVuqYmh0FHYnAeq%2FJi4nqatcwqBqH%2BRiGa3wOl3S%2BLZ0UHQVESQ3zD2ECsgb6fx1v6A%2Fa03Ei8n5BIDMazS7cLQTdgghMzjjusJtutreVQV4ULkcjEy%2BFUcBCUgdpbngJcB57FkPTl59jOrpn81uMEAuL%2Fi%2Fad1Vmj4CzDRl%2BoIn5HFlEdjCtJRFlycTjzYZkfq147fYfMAn2ki43c9CmjJTHj731SAOKIzwAlRu7oBvVtvstg8IpsWYiFWA6BIwc9w9OKX3NPF5qZMf3pmgAa7S3Vac2s%2BN7MjOS2d%2F0GpLtFZ%2B%2BtUW6OTeVhsApoEzEIUuG%2F7LsWlgqlGEh7XBKDIjIS%2FqMln7Pb1MPa03uD3wfDdapU1RZ%2F94SV89IXY1s65O3E4K8%2FSnafInoylzRpB4Ci%2B45p6aKghIhHnYaudLSuSMWVL5g2LL%2FJlMy8MisPta4zGCMK2qU%2BHzp%2F9j%2FNZ2hKqVbd2BPhMSiUlFeaWScemFHyK6fXjCJ7LGcBjqqATehW7Yv5z1x4TmmfIjcoZFQAidoHnwHm9RPXyDbgBTO2udViXEliCUyFsSykOxO8M6c%2B2UbSH9XsyawJb3gmBofydmTsR9B%2F7Ur04Lhoz2oj%2FPmIeVJAgEcknkyjnmYY%2FJsdAuNbUgnDIskdDhikEpLLo%2FfGxrEXGKbe8fRyyOJ8nOboLpBuo%2FeRI6UWyZUExB7bVZwOqyBw1Lv6rJP3mahwxPwM6%2B0hINj&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20221204T114531Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYSY3Z4MUC%2F20221204%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=148fb7fbe9d7a380e9f29fc4f4c21cd9827d1d6cc22b2604dbbc952dcc8bf109&hash=800b97801897343f4a605481c6cf96b5566a2b312d566ec82034a72c784d616c&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0005109812002476&tid=spdf-8919862c-4531-4f9d-b117-88fa5c9bc3bf&sid=b4389eda4ea7f34a4d88f6f8c7034ec25a00gxrqa&type=client&ua=52535b505407020258&rr=77444bdb7cec0489

    :param int forward_step: envmodel forward step.
    :param float gamma: discount factor.
    :param float tau: param for soft update of target network.
    """

    def __init__(self, index=0, **kwargs):
        super().__init__(index, **kwargs)
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.gamma = 0.99
        self.tau = 0.005
        self.pev_step = 1
        self.pim_step = 1
        self.forward_step = 10
        self.tb_info = dict()

    @property
    def adjustable_parameters(self):
        para_tuple = (
            "gamma",
            "tau",
            "pev_step",
            "pim_step",
            "forward_step",
            "reward_scale",
        )
        return para_tuple

    def local_update(self, data: dict, iteration: int) -> dict:
        update_list = self.__compute_gradient(data, iteration)
        self.__update(update_list)
        return self.tb_info

    def get_remote_update_info(self, data: dict, iteration: int) -> Tuple[dict, dict]:
        update_list = self.__compute_gradient(data, iteration)
        update_info = dict()
        for net_name in update_list:
            update_info[net_name] = [
                p.grad for p in self.networks.net_dict[net_name].parameters()
            ]
        return self.tb_info, update_info

    def remote_update(self, update_info: dict):
        for net_name, grads in update_info.items():
            for p, grad in zip(self.networks.net_dict[net_name].parameters(), grads):
                p.grad = grad
        self.__update(list(update_info.keys()))

    def __update(self, update_list):
        tau = self.tau
        for net_name in update_list:
            self.networks.optimizer_dict[net_name].step()

        with torch.no_grad():
            for net_name in update_list:
                for p, p_targ in zip(
                    self.networks.net_dict[net_name].parameters(),
                    self.networks.target_net_dict[net_name].parameters(),
                ):
                    p_targ.data.mul_(1 - tau)
                    p_targ.data.add_(tau * p.data)

    def __compute_gradient(self, data, iteration):
        update_list = []

        start_time = time.time()

        if iteration % (self.pev_step + self.pim_step) < self.pev_step:
            self.networks.v.zero_grad()
            loss_v, v = self.__compute_loss_v(data)
            loss_v.backward()
            self.tb_info[tb_tags["loss_critic"]] = loss_v.item()
            self.tb_info[tb_tags["critic_avg_value"]] = v.item()
            update_list.append("v")
        else:
            self.networks.policy.zero_grad()
            loss_policy = self.__compute_loss_policy(data)
            loss_policy.backward()
            self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()
            update_list.append("policy")

        end_time = time.time()

        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms
        return update_list

    def __compute_loss_v(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        v = self.networks.v(o)
        info_init = data

        with torch.no_grad():
            for step in range(self.forward_step):
                if step == 0:
                    a = self.networks.policy(o)
                    o2, r, d, info = self.envmodel.forward(o, a, d, info_init)
                    backup = r
                else:
                    o = o2
                    a = self.networks.policy(o)
                    o2, r, d, info = self.envmodel.forward(o, a, d, info)
                    backup += self.gamma ** step * r

            backup += (
                (~d) * self.gamma ** self.forward_step * self.networks.v_target(o2)
            )
        loss_v = ((v - backup) ** 2).mean()
        return loss_v, torch.mean(v)

    def __compute_loss_policy(self, data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        info_init = data
        v_pi = torch.zeros(1)
        for p in self.networks.v.parameters():
            p.requires_grad = False
        for step in range(self.forward_step):
            if step == 0:
                a = self.networks.policy(o)
                o2, r, d, info = self.envmodel.forward(o, a, d, info_init)
                v_pi = r
            else:
                o = o2
                a = self.networks.policy(o)
                o2, r, d, info = self.envmodel.forward(o, a, d, info)
                v_pi += self.gamma ** step * r
        v_pi += (~d) * self.gamma ** self.forward_step * self.networks.v_target(o2)
        for p in self.networks.v.parameters():
            p.requires_grad = True
        return -v_pi.mean()


if __name__ == "__main__":
    print("11111")

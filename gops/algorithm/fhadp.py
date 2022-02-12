#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Creator: Fawang ZHANG
#  Update Date: 2021-11-30 create algorithm
#  Comments: ?


__all__ = ['FHADP']
from copy import deepcopy
import torch.nn as nn
from torch.optim import Adam
import time
import warnings
from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.create_pkg.create_env_model import create_env_model
from gops.utils.utils import get_apprfunc_dict
from gops.utils.tensorboard_tools import tb_tags


class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        policy_func_type = kwargs['policy_func_type']
        policy_args = get_apprfunc_dict('policy', policy_func_type, **kwargs)

        self.policy = create_apprfunc(**policy_args)
        self.net_dict = {'policy': self.policy}
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs['policy_learning_rate'])
        self.optimizer_dict = {'policy': self.policy_optimizer}

    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)

    def update(self, grad_info):
        grads_dict = grad_info['grads_dict']
        for net_name, grads in grads_dict.items():
            for p, grad in zip(self.net_dict[net_name].parameters(), grads):
                p.grad = grad
            self.optimizer_dict[net_name].step()

class FHADP:
    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.forward_step = kwargs['pre_horizon']
        self.use_gpu = kwargs['use_gpu']
        if self.use_gpu:
            self.envmodel = self.envmodel.cuda()
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
        params['use_gpu'] = self.use_gpu
        params['reward_scale'] = self.reward_scale
        params['forward_step'] = self.forward_step
        return params

    def compute_gradient(self, data, iteration):
        grad_info = dict()
        grads_dict = dict()
        start_time = time.time()
        if self.use_gpu:
            self.networks = self.networks.cuda()
            for key, value in data.items():
                data[key] = value.cuda()
        self.networks.policy.zero_grad()
        loss_policy = self.compute_loss_policy(deepcopy(data))
        loss_policy.backward()
        policy_grad = [p.grad for p in self.networks.policy.parameters()]
        self.tb_info[tb_tags["loss_actor"]] = loss_policy.item()
        grads_dict['policy'] = policy_grad

        if self.use_gpu:
            self.networks = self.networks.cpu()
            for key, value in data.items():
                data[key] = value.cpu()

        end_time = time.time()

        self.tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

        grad_info['grads_dict'] = grads_dict
        return grad_info, self.tb_info

    def compute_loss_policy(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']  # TODO  解耦字典

        # v_pi = torch.zeros(1)
        # for step in range(self.forward_step):
        #     if step == 0:
        #         a = self.networks.policy(o)
        #         o2, r, d = self.envmodel.forward(o, a, d)
        #         v_pi = r
        #     else:
        #         o = o2
        #         a = self.networks.policy(o)
        #         o2, r, d = self.envmodel.forward(o, a, d)
        #         v_pi += r
        # print(v_pi.shape,v_pi.type(),-v_pi.mean())
        # return -v_pi.mean()

        next_state_list, v_pi, done_list = self.envmodel.forward_n_step(o, self.networks.policy, self.forward_step, d)
        return -(v_pi*self.reward_scale).mean()

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)


if __name__ == '__main__':
    print('11111')

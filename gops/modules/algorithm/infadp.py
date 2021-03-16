#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao
#  Update Date: 2020-11-13
#  Update Date: 2021-01-03
#  Comments: ?


__all__ = ['INFADP']

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam

from modules.create_pkg.create_apprfunc import create_apprfunc
from modules.create_pkg.create_env_model import create_env_model
from modules.utils.utils import get_apprfunc_dict


class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.polyak = 1 - kwargs['tau']
        self.delay_update = kwargs['delay_update']
        v_args = get_apprfunc_dict('value', **kwargs)
        self.v = create_apprfunc(**v_args)

        policy_args = get_apprfunc_dict('policy', **kwargs)
        self.policy = create_apprfunc(**policy_args)

        # self.v_target = deepcopy(self.v)
        # self.policy_target = deepcopy(self.policy)

        '''for p in self.v_target.parameters():
            p.requires_grad = False
        for p in self.policy_target.parameters():
            p.requires_grad = False
        '''
        self.policy_optimizer = Adam(self.policy.parameters(), lr=kwargs['policy_learning_rate'])  #
        self.v_optimizer = Adam(self.v.parameters(), lr=kwargs['value_learning_rate'])

    def update(self, grads, iteration):
        v_grad_len = len(list(self.v.parameters()))
        v_grad, policy_grad = grads[:v_grad_len], grads[v_grad_len:]
        for p, grad in zip(self.v.parameters(), v_grad):
            p._grad = torch.from_numpy(grad)
        for p, grad in zip(self.policy.parameters(), policy_grad):
            p._grad = torch.from_numpy(grad)
        self.v_optimizer.step()
        if iteration % self.delay_update == 0:
            self.policy_optimizer.step()
        '''with torch.no_grad():
            for p, p_targ in zip(self.v.parameters(), self.v_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(self.policy.parameters(), self.policy_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        '''


class INFADP():
    def __init__(self, **kwargs):
        self.networks = ApproxContainer(**kwargs)
        self.envmodel = create_env_model(**kwargs)
        self.gamma = kwargs['gamma']
        self.forward_step = 10
        self.reward_scale = kwargs['reward_scale']
        # self.polyak = 1 - kwargs['tau']
        self.policy_optimizer = Adam(self.networks.policy.parameters(), lr=kwargs['policy_learning_rate'])  #
        self.v_optimizer = Adam(self.networks.v.parameters(), lr=kwargs['value_learning_rate'])
        torch.autograd.set_detect_anomaly(True)

    def compute_gradient(self, data):
        self.v_optimizer.zero_grad()
        loss_v = self.compute_loss_v(deepcopy(data))
        loss_v.backward()

        # for p in self.networks.q.parameters():
        #     p.requires_grad = False

        self.policy_optimizer.zero_grad()
        loss_policy = self.compute_loss_policy(deepcopy(data))
        loss_policy.backward()

        # for p in self.networks.q.parameters():
        #     p.requires_grad = True
        v_grad = [p._grad.numpy() for p in self.networks.v.parameters()]
        policy_grad = [p._grad.numpy() for p in self.networks.policy.parameters()]
        return v_grad + policy_grad

    def compute_loss_v(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']  # TODO  解耦字典
        v = self.networks.v(o)

        with torch.no_grad():
            for step in range(self.forward_step):
                if step == 0:
                    a = self.networks.policy(o)
                    o2, r, d, _ = self.envmodel.step(o, a)
                    backup = self.reward_scale * r
                else:
                    o = o2.detach()
                    a = self.networks.policy(o)
                    o2, r, d, _ = self.envmodel.step(o, a)
                    backup += self.reward_scale * self.gamma ** step * r

            backup += (~d) * self.gamma ** (self.forward_step) * self.networks.v(o2)
        loss_v = ((v - backup) ** 2).mean()
        return loss_v

    def compute_loss_policy(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']  # TODO  解耦字典
        v_pi = torch.zeros(1)
        for p in self.networks.v.parameters():
            p.requires_grad = False
        for step in range(self.forward_step):
            if step == 0:
                a = self.networks.policy(o)
                # a.retain_grad()
                o2, r, d, _ = self.envmodel.step(o, a)
                # o2.retain_grad()

                # print(a.grad)
                v_pi = self.reward_scale * r
            else:
                o.copy_(o2.detach())
                a = self.networks.policy(o)
                o2, r, d, _ = self.envmodel.step(o, a)
                v_pi += self.reward_scale * self.gamma ** step * r
        v_pi += self.gamma ** (self.forward_step) * self.networks.v(o2)
        for p in self.networks.v.parameters():
            p.requires_grad = True
        return -v_pi.mean()


if __name__ == '__main__':
    print('11111')
    import mujoco_py

    print('11111')
    import os

    print('11111')
    mj_path, _ = mujoco_py.utils.discover_mujoco()
    print('11111')
    xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
    print('11111')
    model = mujoco_py.load_model_from_path(xml_path)
    print('11111')
    sim = mujoco_py.MjSim(model)

    print(sim.data.qpos)
    # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

    sim.step()
    print(sim.data.qpos)

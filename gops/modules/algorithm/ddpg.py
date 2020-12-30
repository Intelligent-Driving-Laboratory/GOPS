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
#  Comments: ?


__all__ = ['DDPG']

from copy import deepcopy
import torch
import torch.nn as nn
from torch.optim import Adam
from ocp_tools.modules.create_pkg.create_apprfunc import create_apprfunc



class AC_ApprFunc(nn.Module):
    def __init__(self, actor,critic):
        super().__init__()
        self.pi = actor
        self.q = critic

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class DDPG():
    def __init__(self,pi_lr=0.001,q_lr=0.001,gamma=0.99,polyak=0.995,**kwargs):
        # 创建网络：和算法相关，返回：算法所有的网近似函数
        # 多进程，加载其它网络参数：输入：字典 key，net（参数） 返回：null

        #print(kwargs['action_high_limit'][0])

        actor_dict = {'apprfunc':kwargs['apprfunc'],
                      'name':kwargs['policy_func_name'],
                      'obs_dim':kwargs['obsv_dim'],
                      'act_dim':kwargs['action_dim'],
                      'hidden_sizes':(kwargs['policy_hidden_units'],kwargs['policy_hidden_units']),
                      'activation': kwargs['policy_output_activation'],
                      'act_limit': 1
                      }

        critic_dict = {'apprfunc':kwargs['apprfunc'],
                       'name':kwargs['value_func_name'],
                       'obs_dim': kwargs['obsv_dim'],
                       'act_dim': kwargs['action_dim'],
                       'hidden_sizes': (kwargs['value_hidden_units'],kwargs['value_hidden_units']),
                       'activation': kwargs['value_output_activation'],
                       }

        actor = create_apprfunc(**actor_dict) #(name,**dict)
        critic = create_apprfunc(**critic_dict)

        self.model = AC_ApprFunc(actor,critic)  # TODO change name: model->apprfunc

        self.target_model =  deepcopy(self.model)
        for p in self.target_model.parameters():
            p.requires_grad = False

        # 配置q与pi网络的优化器
        self.gamma = gamma
        self.polyak = polyak
        self.pi_optimizer = Adam(self.model.pi.parameters(), lr=pi_lr) #
        self.q_optimizer = Adam(self.model.q.parameters(), lr=q_lr)


    def learn(self,data):
        #
        self.q_optimizer.zero_grad()
        loss_q = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        #
        for p in self.model.q.parameters():
            p.requires_grad = False

        #
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        #
        for p in self.model.q.parameters():
            p.requires_grad = True
        #
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.target_model.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)


    def predict(self,obs):
        return self.model.act(obs)

    def compute_loss_q(self,data):  # TODO change name: q->Q
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q = self.model.q(o,a)

        with torch.no_grad():
            q_pi_targ = self.target_model.q(o2, self.target_model.pi(o2))
            backup = r +  self.gamma * (1 - d) * q_pi_targ

        loss_q = ((q - backup)**2).mean()
        return loss_q

    def compute_loss_pi(self,data): # TODO change name: pi->policy
        o = data['obs']
        q_pi = self.model.q(o, self.model.pi(o))
        return -q_pi.mean()

    def getEnvModel(self,func):
        pass

if __name__ == '__main__':
    pass

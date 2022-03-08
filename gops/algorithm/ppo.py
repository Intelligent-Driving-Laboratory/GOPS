#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Proximal Policy Optimization Algorithm (PPO)
#  Update: 2021.03.05, Shengbo LI (example, can be deleted)


__all__ = ['ApproxContainer', 'PPO']


from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import time
import warnings

from gops.create_pkg.create_apprfunc import create_apprfunc
from gops.utils.utils import get_apprfunc_dict
from gops.utils.tensorboard_tools import tb_tags


class ApproxContainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        value_func_type = kwargs['value_func_type']
        policy_func_type = kwargs['policy_func_type']

        if kwargs['cnn_shared']:  # todo:设置默认false
            feature_args = get_apprfunc_dict('feature', value_func_type, **kwargs)
            kwargs['feature_net'] = create_apprfunc(**feature_args)

        value_args = get_apprfunc_dict('value', value_func_type, **kwargs)
        self.value = create_apprfunc(**value_args)
        policy_args = get_apprfunc_dict('policy', policy_func_type, **kwargs)
        self.policy = create_apprfunc(**policy_args)

    # create action_distributions
    def create_action_distributions(self, logits):
        return self.policy.get_act_dist(logits)

    def update(self, info: dict):
        value_weights = info['value_weights']
        policy_weights = info['policy_weights']

        for p, weight in zip(self.value.parameters(), value_weights):
            p.data = torch.from_numpy(weight)

        for p, weight in zip(self.policy.parameters(), policy_weights):
            p.data = torch.from_numpy(weight)

class PPO():
    def __init__(self, **kwargs):
        self.trainer_type = kwargs['trainer']
        self.max_iteration = kwargs['max_iteration']
        self.num_epoch = kwargs['num_epoch']
        self.num_repeat = kwargs['num_repeat']
        self.num_mini_batch = kwargs['num_mini_batch']
        self.mini_batch_size = kwargs['mini_batch_size']
        self.sample_batch_size = kwargs['sample_batch_size']
        self.indices = np.arange(self.sample_batch_size)

        # Parameters for algorithm
        self.gamma = 0.99
        self.lamb = 0.95  # applied in GAE, making a compromise between bias & var
        self.clip = 0.2
        self.clip_now = self.clip
        self.EPS = 1e-8
        self.loss_coefficient_kl = 0.2
        self.loss_coefficient_value = 1.0
        self.loss_coefficient_entropy = 0.0

        self.schedule_adam = 'none'
        self.schedule_clip = 'none'
        self.advantage_norm = True
        self.loss_value_clip = True
        self.value_clip = 10.0
        self.loss_value_norm = False

        self.networks = ApproxContainer(**kwargs)
        self.learning_rate = kwargs['learning_rate']
        self.approximate_optimizer = Adam(self.networks.parameters(), lr=self.learning_rate)
        self.use_gpu = kwargs['use_gpu']
        if self.use_gpu:
            self.networks.value = self.networks.value.cuda()
            self.networks.policy = self.networks.policy.cuda()

    def set_parameters(self, param_dict):
        for key in param_dict:
            if hasattr(self, key):
                setattr(self, key, param_dict[key])
            else:
                warning_msg = "param '" + key + "'is not defined in algorithm!"
                warnings.warn(warning_msg)
        print('--------------------------------')
        print('| Proximal Policy Optimization |')
        print('| {:<16}'.format('gamma') + ' | ' + '{:<9} |'.format(self.gamma))
        print('| {:<16}'.format('lambda') + ' | ' + '{:<9} |'.format(self.lamb))
        print('| {:<16}'.format('clip') + ' | ' + '{:<9} |'.format(str(self.clip)))
        print('| {:<16}'.format('factor_value') + ' | ' + '{:<9} |'.format(str(self.loss_coefficient_value)))
        print('| {:<16}'.format('factor_entropy') + ' | ' + '{:<9} |'.format(str(self.loss_coefficient_entropy)))
        print('| {:<16}'.format('factor_kl') + ' | ' + '{:<9} |'.format(str(self.loss_coefficient_kl)))
        print('| {:<16}'.format('schedule_adam') + ' | ' + '{:<9} |'.format(self.schedule_adam))
        print('| {:<16}'.format('schedule_clip') + ' | ' + '{:<9} |'.format(self.schedule_clip))
        print('| {:<16}'.format('advantage_norm') + ' | ' + '{:<9} |'.format(str(self.advantage_norm)))
        print('| {:<16}'.format('loss_value_clip') + ' | ' + '{:<9} |'.format(str(self.loss_value_clip)))
        print('| {:<16}'.format('loss_value_norm') + ' | ' + '{:<9} |'.format(str(self.loss_value_norm)))
        print('--------------------------------')

    def get_parameters(self):
        params = dict()
        params['is_gpu'] = self.use_gpu
        params['gamma'] = self.gamma
        params['lamb'] = self.lamb
        params['clip'] = self.clip
        params['clip_now'] = self.clip_now
        params['EPS'] = self.EPS
        params['loss_coefficient_kl'] = self.loss_coefficient_kl
        params['loss_coefficient_value'] = self.loss_coefficient_value
        params['loss_coefficient_entropy'] = self.loss_coefficient_entropy

        params['schedule_adam'] = self.schedule_adam
        params['schedule_clip'] = self.schedule_clip
        params['advantage_norm'] = self.advantage_norm
        params['loss_value_clip'] = self.loss_value_clip
        params['value_clip'] = self.value_clip
        params['loss_value_norm'] = self.loss_value_norm

        return params

    def compute_gradient(self, data: Dict[str, torch.Tensor], iteration: int):
        start_time = time.perf_counter()

        data_gae = self.__generalization_advantage_estimate(data)  # 1/10 of total time

        for _ in range(self.num_repeat):
            np.random.shuffle(self.indices)

            for n in range(self.num_mini_batch):
                mb_start = self.mini_batch_size * n
                mb_end = self.mini_batch_size * (n + 1)
                mb_indices = self.indices[mb_start:mb_end]
                mb_sample = {k: v[mb_indices] for k, v in data_gae.items()}
                loss_total, loss_surrogate, loss_value, loss_entropy, approximate_kl, clip_fra = \
                    self.__compute_loss(mb_sample, iteration)
                self.approximate_optimizer.zero_grad()
                loss_total.backward()
                self.approximate_optimizer.step()
                if self.schedule_adam == 'linear':
                    decay_rate = 1 - (iteration / self.max_iteration)
                    assert decay_rate >= 0, "the decay_rate is less than 0!"
                    lr_now = self.learning_rate * decay_rate
                    # set learning rate
                    for g in self.approximate_optimizer.param_groups:
                        g['lr'] = lr_now

        value_weights = [p.detach().cpu().numpy() for p in self.networks.value.parameters()]
        policy_weights = [p.detach().cpu().numpy() for p in self.networks.policy.parameters()]

        end_time = time.perf_counter()
        
        tb_info = dict()
        # tb_info[tb_tags["loss_total"]] = loss_total.item()
        tb_info[tb_tags["loss_actor"]] = loss_surrogate.item()
        tb_info[tb_tags["loss_critic"]] = loss_value.item()
        # tb_info[tb_tags["loss_entropy"]] = loss_entropy.item()
        tb_info["Train/kl_divergence"] = approximate_kl.item()
        # tb_info[tb_tags["clip_fraction"]] = clip_fra.item()
        tb_info[tb_tags["alg_time"]] = (end_time - start_time) * 1000  # ms

        grad_info = dict()
        grad_info['value_weights'] = value_weights
        grad_info['policy_weights'] = policy_weights
        grad_info['iteration'] = iteration

        return grad_info, tb_info

    def __compute_loss(self, data: Dict[str, torch.Tensor], iteration: int):
        obs, act, rew, obs2 = data['obs'], data['act'], data['rew'], data['obs2']
        mask, pro = data['mask'], data['pro']
        returns, advantages, values = data['ret'], data['adv'], data['val']
        logits = data['logits']

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
        loss_surrogate = - torch.mean(torch.min(sur1, sur2))

        if self.loss_value_clip:  # reduce variability during critic training, but increase the loss_critic
            # unclipped value
            value_losses1 = torch.pow(mb_new_value - mb_return, 2)
            # clipped value
            mb_new_value_clipped = mb_old_value + (mb_new_value - mb_old_value).clamp(-self.value_clip, self.value_clip)
            value_losses2 = torch.pow(mb_new_value_clipped - mb_return, 2)
            # value loss
            value_losses = torch.max(value_losses1, value_losses2)
        else:
            value_losses = torch.pow(mb_new_value - mb_return, 2)
        if self.loss_value_norm:
            mb_return_6std = 6 * mb_return.std()
            loss_value = torch.mean(value_losses) / mb_return_6std
        else:
            loss_value = 0.5 * torch.mean(value_losses)

        # entropy loss
        loss_entropy = torch.mean(mb_new_act_dist.entropy())
        loss_kl = torch.mean(mb_old_act_dist.kl_divergence(mb_new_act_dist))
        clip_fraction = torch.mean(torch.gt(torch.abs(ratio - 1.0), self.clip_now).float())

        # total loss
        loss_total = (
            loss_surrogate +
            self.loss_coefficient_kl * loss_kl +
            self.loss_coefficient_value * loss_value -
            self.loss_coefficient_entropy * loss_entropy
        )

        if self.schedule_clip == 'linear':
            decay_rate = 1 - (iteration / self.max_iteration)
            assert decay_rate >= 0, "the decay_rate is less than 0!"
            self.clip_now = self.clip * decay_rate

        return loss_total, loss_surrogate, loss_value, loss_entropy, loss_kl, clip_fraction

    def __generalization_advantage_estimate(self, data: Dict[str, torch.Tensor]):
        if self.use_gpu:
            obs, act, rew, obs2, done = data['obs'].cuda(), data['act'].cuda(), data['rew'].cuda(), data['obs2'].cuda(), data['done'].cuda()
            logp, time_limited = data['logp'].cuda(), data['time_limited'].cuda()
        else:
            obs, act, rew, obs2, done = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
            logp, time_limited = data['logp'], data['time_limited']
        with torch.no_grad():
            pro = logp
            logits = self.networks.policy(obs)
            values = self.networks.value(obs)
            prev_value = self.networks.value(obs2[-1].unsqueeze(0))

        mask = (~done.to(torch.bool) | time_limited.to(torch.bool)).to(torch.int)  # useless?
        deltas = torch.zeros_like(done)
        advantages = torch.zeros_like(done)

        prev_advantage = 0
        for i in reversed(range(self.sample_batch_size)):
            # generalization advantage estimate, ref: https://arxiv.org/pdf/1506.02438.pdf
            deltas[i] = rew[i] + self.gamma * prev_value * (1 - done[i]) - values[i]
            advantages[i] = deltas[i] + self.gamma * self.lamb * prev_advantage * (1 - done[i])

            prev_value = values[i]
            prev_advantage = advantages[i]
        returns = advantages + values

        if self.advantage_norm:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.EPS)  # standardization

        return dict(obs=obs, act=act, rew=rew, obs2=obs2, logits=logits,
                    mask=mask, pro=pro, ret=returns, adv=advantages, val=values)

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)


if __name__ == '__main__':
    print('this is PPO algorithm!')
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

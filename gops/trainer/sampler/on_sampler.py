#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yang GUAN
#  Description: Monte Carlo Sampler
#
#  Update Date: 2021-03-10, Wenhan CAO: Revise Codes


import numpy as np
import torch

from gops.create_pkg.create_env import create_env
from gops.utils.action_distributions import GaussDistribution, DiracDistribution, ValueDiracDistribution, CategoricalDistribution
from gops.utils.noise import GaussNoise, EpsilonGreedy
import time
from gops.utils.tensorboard_tools import tb_tags
from gops.utils.utils import array_to_scalar


class OnSampler():
    def __init__(self, **kwargs):
        self.env = create_env(**kwargs)
        alg_name = kwargs['algorithm']
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)
        self.noise_params = kwargs['noise_params']
        self.sample_batch_size = kwargs['sample_batch_size']
        self.obs = self.env.reset()
        self.has_render = hasattr(self.env, 'render')
        self.policy_func_name = kwargs['policy_func_name']
        self.action_type = kwargs['action_type']
        self.total_sample_number = 0
        self.obsv_dim = kwargs['obsv_dim']
        self.act_dim = kwargs['action_dim']
        if 'constraint_dim' in kwargs.keys():
            self.is_constrained = True
            self.con_dim = kwargs['constraint_dim']
        else:
            self.is_constrained = False
        if 'adversary_dim' in kwargs.keys():
            self.is_adversary = True
            self.advers_dim = kwargs['adversary_dim']
        else:
            self.is_adversary = False
        if self.action_type == 'continu':
            self.noise_processor = GaussNoise(**self.noise_params)
            if self.policy_func_name == 'StochaPolicy':
                self.action_distirbution_cls = GaussDistribution
            elif self.policy_func_name == 'DetermPolicy':
                self.action_distirbution_cls = DiracDistribution
        elif self.action_type == 'discret':
            self.noise_processor = EpsilonGreedy(**self.noise_params)
            if self.policy_func_name == 'StochaPolicyDis':
                self.action_distirbution_cls = CategoricalDistribution
            elif self.policy_func_name == 'DetermPolicyDis':
                self.action_distirbution_cls = ValueDiracDistribution

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def sample(self):
        self.total_sample_number += self.sample_batch_size
        tb_info = dict()
        start_time = time.perf_counter()
        batch_data = []
        for _ in range(self.sample_batch_size):
            batch_obs = torch.from_numpy(np.expand_dims(self.obs, axis=0).astype('float32'))
            if self.action_type == 'continu':
                logits = self.networks.policy(batch_obs)
            else:
                logits = self.networks.policy.q(batch_obs)

            action_distribution = self.action_distirbution_cls(logits)
            action = action_distribution.sample().detach()[0]
            if hasattr(action_distribution, 'log_prob'):
                logp = action_distribution.log_prob(action).item()
            else:
                logp = 0.
            action = action.numpy()
            if self.noise_params is not None:
                action = self.noise_processor.sample(action)
            action = np.array(action)  # ensure action is an array
            next_obs, reward, self.done, info = self.env.step(action)
            if 'TimeLimit.truncated' not in info.keys():
                info['TimeLimit.truncated'] = False
            if info['TimeLimit.truncated']:
                self.done = False
            data = [self.obs.copy(), action, reward, next_obs.copy(), self.done, logp, info['TimeLimit.truncated']]
            if self.is_constrained:
                constraint = info['constraint']
            else:
                constraint = None
            if self.is_adversary:
                sth_about_adversary = np.zeros(self.advers_dim)
            else:
                sth_about_adversary = None
            data.append(constraint)
            data.append(sth_about_adversary)
            batch_data.append(tuple(data))
            self.obs = next_obs
            if self.done or info['TimeLimit.truncated']:
                self.obs = self.env.reset()

        end_time = time.perf_counter()
        tb_info[tb_tags["sampler_time"]] = (end_time - start_time) * 1000

        return batch_data, tb_info

    def get_total_sample_number(self):
        return self.total_sample_number

    def samples_conversion(self, samples):
        obs_dim = self.obsv_dim
        if isinstance(obs_dim, int):
            obs_dim = (obs_dim,)
        tensor_dict = {'obs': torch.zeros((self.sample_batch_size,) + obs_dim),
                       'act': torch.zeros(self.sample_batch_size, self.act_dim),
                       'obs2': torch.zeros((self.sample_batch_size,) + obs_dim),
                       'rew': torch.zeros(self.sample_batch_size, ),
                       'done': torch.zeros(self.sample_batch_size, ),
                       'logp': torch.zeros(self.sample_batch_size, ),
                       'time_limited': torch.zeros(self.sample_batch_size, )}
        if self.is_constrained:
            tensor_dict['con'] = torch.zeros(self.sample_batch_size, self.con_dim)
        else:
            tensor_dict['con'] = None
        if self.is_adversary:
            tensor_dict['advers'] = torch.zeros(self.sample_batch_size, self.advers_dim)
        else:
            tensor_dict['advers'] = None
        idx = 0
        for sample in samples:
            obs, act, rew, next_obs, done, logp, time_limited = sample[:7]
            tensor_dict['obs'][idx] = torch.from_numpy(obs)
            tensor_dict['act'][idx] = torch.from_numpy(act)
            tensor_dict['rew'][idx] = torch.tensor(array_to_scalar(rew))
            tensor_dict['obs2'][idx] = torch.from_numpy(next_obs)
            tensor_dict['done'][idx] = torch.tensor(array_to_scalar(done))
            tensor_dict['logp'][idx] = torch.tensor(logp)
            tensor_dict['time_limited'][idx] = torch.tensor(time_limited)
            if self.is_constrained:
                con = sample[7]
                tensor_dict['con'][idx] = torch.from_numpy(con)
            if self.is_adversary:
                advers = sample[-1]
                tensor_dict['advers'][idx] = torch.from_numpy(advers)
            idx += 1
        tensor_dict['time_limited'][-1] = True
        return tensor_dict

    def sample_with_replay_format(self):
        samples, sampler_tb_dict = self.sample()
        return self.samples_conversion(samples), sampler_tb_dict

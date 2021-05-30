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

from modules.create_pkg.create_env import create_env
from modules.utils.action_distributions import GaussDistribution, DiracDistribution, ValueDiracDistribution
from modules.utils.noise import GaussNoise, EpsilonGreedy
import time
from modules.utils.tensorboard_tools import tb_tags


class McSampler():
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

        if self.action_type == 'continu':
            self.noise_processor = GaussNoise(**self.noise_params)
            if self.policy_func_name == 'StochaPolicy':
                self.action_distirbution_cls = GaussDistribution
            elif self.policy_func_name == 'DetermPolicy':
                self.action_distirbution_cls = DiracDistribution
        elif self.action_type == 'discret':
            self.noise_processor = EpsilonGreedy(**self.noise_params)
            self.action_distirbution_cls = ValueDiracDistribution

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def sample(self):
        tb_info = dict()
        start_time = time.time()
        batch_data = []
        for _ in range(self.sample_batch_size):
            batch_obs = torch.from_numpy(np.expand_dims(self.obs, axis=0).astype('float32'))
            if self.action_type == 'continu':
                logits = self.networks.policy(batch_obs)
            else:
                logits = self.networks.policy.q(batch_obs)

            action_distribution = self.action_distirbution_cls(logits)
            action = action_distribution.sample().detach().numpy()[0]
            if hasattr(action_distribution, 'log_prob'):
                logp = action_distribution.log_prob(action).detach().numpy()[0]
            else:
                logp = 0.

            if self.noise_params is not None:
                action = self.noise_processor.sample(action)

            next_obs, reward, self.done, info = self.env.step(action)
            batch_data.append(
                (self.obs.copy(), action, reward, next_obs.copy(), self.done))
            self.obs = next_obs
            if self.done:
                self.obs = self.env.reset()
        end_time = time.time()
        tb_info[tb_tags["sampler_time"]] = (end_time - start_time) * 1000

        return batch_data, tb_info

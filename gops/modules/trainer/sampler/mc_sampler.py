#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Yang GUAN, Wenhan CAO

import numpy as np
import torch

from modules.create_pkg.create_env import create_env
from modules.utils.action_distributions import GaussDistribution, DiracDistribution, ValueDiracDistribution
from modules.utils.noise import GaussNoise, EpsilonGreedy


class McSampler():
    def __init__(self, **kwargs):
        self.env = create_env(**kwargs)
        alg_name = kwargs['algorithm']
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)
        self.batch_size = kwargs['batch_size']
        self.render = kwargs['is_render']
        self.reward_scale = kwargs['reward_scale']
        self.noise_params = kwargs['noise_params']
        self.sample_batch_size = kwargs['sample_batch_size']
        self.obs = self.env.reset()
        self.distribution_type = kwargs['distribution_type']
        self.has_render = hasattr(self.env, 'render')
        self.action_type = kwargs['action_type']

        if self.distribution_type == 'Dirac':
            self.action_distirbution_cls = DiracDistribution
        elif self.distribution_type == 'Gauss':
            self.action_distirbution_cls = GaussDistribution
        elif self.distribution_type == 'ValueDirac':
            self.action_distirbution_cls = ValueDiracDistribution

        if self.noise_params is not None:
            if self.action_type == 'conti':
                self.noise_processor = GaussNoise(**self.noise_params)
            else:
                self.noise_processor = EpsilonGreedy(**self.noise_params)
        else:
            if self.action_type == 'conti':
                self.noise_processor = GaussNoise()
            else:
                self.noise_processor = EpsilonGreedy()


    def sample(self):
        batch_data = []
        for _ in range(self.sample_batch_size):
            batch_obs = torch.from_numpy(np.expand_dims(self.obs, axis=0).astype('float32'))
            if self.action_type == 'conti':
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
            #if self.render:
             #   self.env.render()
            batch_data.append(
                (self.obs.copy(), action, reward * self.reward_scale, next_obs.copy(), self.done))  # TODO  加入logp
            self.obs = next_obs
            if self.done:
                self.obs = self.env.reset()
        return batch_data

#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Yang GUAN

import datetime
import os
import logging
import time

import numpy as np
import torch
from modules.create_pkg.create_env import create_env
from torch.utils.tensorboard import SummaryWriter
from modules.utils.action_distributions import GaussDistribution, DiracDistribution, ValueDiracDistribution
from modules.utils.tensorboard_tools import tb_tags
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator():

    def __init__(self, **kwargs):
        self.env = create_env(**kwargs)
        alg_name = kwargs['algorithm']
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)
        self.render = kwargs['is_render']
        self.save_folder = kwargs['save_folder']  # TODO get parent dir
        self.num_eval_episode = kwargs['num_eval_episode']
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
        self.writer.add_scalar(tb_tags['time'], 0, 0)

        self.writer.flush()

        self.distribution_type = kwargs['distribution_type']
        if self.distribution_type == 'Dirac':
            self.action_distirbution_cls = DiracDistribution
        elif self.distribution_type == 'Gauss':
            self.action_distirbution_cls = GaussDistribution
        elif self.distribution_type == 'ValueDirac':
            self.action_distirbution_cls = ValueDiracDistribution

    def run_an_episode(self, render=True):
        reward_list = []
        obs = self.env.reset()
        done = 0
        while not done:
            batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype('float32'))
            logits = self.networks.policy(batch_obs)
            action_distribution = self.action_distirbution_cls(logits)
            action = action_distribution.mode()
            action = action.detach().numpy()[0]
            next_obs, reward, done, info = self.env.step(action)
            obs = next_obs
            if render: self.env.render()
            reward_list.append(reward)
        episode_return = sum(reward_list)
        return episode_return

    def run_n_episodes(self, n):
        episode_return_list = []
        for _ in range(n):
            episode_return_list.append(self.run_an_episode(self.render))
        return np.mean(episode_return_list)

    def run_evaluation(self, iteration):
        self.writer.add_scalar(tb_tags['total_average_return'], self.run_n_episodes(self.num_eval_episode), iteration)

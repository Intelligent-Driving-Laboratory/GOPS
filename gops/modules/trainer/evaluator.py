#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yang GUAN
#  Description: Evaluation of trained policy
#
#  Update Date: 2021-05-10, Shengbo LI: renew env para

import datetime
import os

import time

import numpy as np
import torch
from modules.create_pkg.create_env import create_env

from modules.utils.action_distributions import GaussDistribution, DiracDistribution, ValueDiracDistribution



class Evaluator():

    def __init__(self, **kwargs):
        self.env = create_env(**kwargs)
        alg_name = kwargs['algorithm']
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, 'ApproxContainer')
        self.networks = ApproxContainer(**kwargs)
        self.render = kwargs['is_render']

        self.num_eval_episode = kwargs['num_eval_episode']
        self.action_type = kwargs['action_type']
        self.policy_func_name = kwargs['policy_func_name']

        if self.action_type == 'continu':
            if self.policy_func_name == 'StochaPolicy':
                self.action_distirbution_cls = GaussDistribution
            elif self.policy_func_name == 'DetermPolicy':
                self.action_distirbution_cls = DiracDistribution
        elif self.action_type == 'discret':
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
            # Draw environment animation
            if render:
                self.env.render()
                # draw action curves - TensorBoard
            reward_list.append(reward)
        episode_return = sum(reward_list)
        return episode_return

    def run_n_episodes(self, n):
        episode_return_list = []
        for _ in range(n):
            episode_return_list.append(self.run_an_episode(self.render))
        return np.mean(episode_return_list)

    def run_evaluation(self):
        return self.run_n_episodes(self.num_eval_episode)

        # add self.writer:

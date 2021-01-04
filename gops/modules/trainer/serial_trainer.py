#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao


__all__ = ['SerialTrainer']

import numpy as np
import torch
import tensorboardX # TODO save data and visualization

from modules.create_pkg.create_buffer import create_buffer


class SerialTrainer():
    def __init__(self,alg,env,**kwargs):
        self.algo = alg
        self.env = env

        self.batch_size = kwargs['batch_size']
        self.render = kwargs['is_render']
        self.warm_size = kwargs['buffer_warm_size']
        self.reward_scale = kwargs['reward_scale']
        self.max_train_episode = kwargs['max_train_episode']
        self.episode_len = kwargs['episode_length']
        self.noise = kwargs['noise']

        self.has_render = hasattr(env,'render')
        self.buffer = create_buffer(**kwargs)


    def run_episode(self):
        obs = self.env.reset()
        total_reward = 0
        steps = 0
        while True:
            steps += 1
            batch_obs = np.expand_dims(obs, axis=0)
            action = self.algo.predict(torch.from_numpy(batch_obs.astype('float32')))
            # 增加探索扰动, 输出限制在 [-1.0, 1.0] 范围内
            # action = np.clip(np.random.normal(action, self.noise), -1.0, 1.0) # todo add train nosie
            next_obs, reward, done, info = self.env.step(action)
            action = [action]
            # store in buffer
            self.buffer.store(obs, action, self.reward_scale * reward, next_obs, done)
            # buffer size > warm size
            if self.buffer.size > self.warm_size and (steps % 5) == 0:
                batch = self.buffer.sample_batch(self.batch_size)
                self.algo.learn(data = batch)

            obs = next_obs
            total_reward += reward

            if done or steps >= self.episode_len:
                break

        return total_reward

    def train(self):
        # store data in buffer
        while self.buffer.size < self.warm_size:
            self.run_episode()

        episode = 0
        total_reward = 0
        while episode < self.max_train_episode:
            for i in range(50):
                total_reward = self.run_episode()
                episode += 1

            # log save

            # apprfunc save

            # eval and render
            eval_reward = self.eval(self.render)
            print("episode =", episode ,",training reward = ",total_reward,",eval reward = ",eval_reward)


    def eval(self,is_render=True):
        obs = self.env.reset()
        total_reward = 0
        steps = 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = self.algo.predict(torch.from_numpy(batch_obs.astype('float32')))
            action = np.clip(action, -1.0, 1.0)

            steps += 1
            next_obs, reward, done, info = self.env.step(action)

            obs = next_obs
            total_reward += reward

            if is_render and self.has_render :
                self.env.render()

            if done or steps >= self.episode_len:
                break

        return total_reward

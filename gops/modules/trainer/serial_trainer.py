#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Sun Hao
#  Update Date: 2021-01-03, Yuxuan JIANG & Guojian ZHAN : modified to allow discrete action space


__all__ = ['SerialTrainer']

import numpy as np
import torch
from modules.create_pkg.create_buffer import create_buffer


class SerialTrainer():
    def __init__(self,alg,env,NOISE=0.05,REWARD_SCALE=0.1,BATCH_SIZE=32,TRAIN_EPISODE=5000,IS_RENDER=False,**kwargs):
        self.algo = alg
        self.env = env

        self.noise = NOISE
        self.reward_scale = REWARD_SCALE
        self.batch_size = BATCH_SIZE
        self.train_eposode = TRAIN_EPISODE
        self.render = IS_RENDER

        self.buffer = create_buffer(**kwargs)
        self.warm_size = kwargs['buffer_warm_size']
        # 创建网络
        # 定义优化器

        # Default to conti for non-breaking change
        self.is_conti = kwargs.get('action_type', 'conti') == 'conti'
        self.global_steps = 0

        if not self.is_conti:
            # TODO: Pass arguments to EpsilonScheduler.
            self.epsilon_scheduler = EpsilonScheduler()

        # Store additional arguments
        self.additional_args = kwargs

    def run_episode(self):
        obs = self.env.reset()
        total_reward = 0
        steps = 0
        while True:
            steps += 1
            self.global_steps += 1
            batch_obs = np.expand_dims(obs, axis=0)
            action = self.algo.predict(torch.from_numpy(batch_obs.astype('float32')))
            # 增加探索扰动, 输出限制在 [-1.0, 1.0] 范围内
            action = self.process_action(action, noise=True)
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

            #print(self.algo.rpm.size)
            if done or steps >= 200:
                break
        return total_reward

    def evaluate(self):
        eval_reward = []
        for i in range(5):
            obs = self.env.reset()
            total_reward = 0
            steps = 0
            while True:
                batch_obs = np.expand_dims(obs, axis=0)
                action = self.algo.predict(batch_obs.astype('float32'))
                action = self.process_action(action, noise=False)

                steps += 1
                next_obs, reward, done, info = self.env.step(action)

                obs = next_obs
                total_reward += reward

                if self.render:
                    self.env.render()
                if done or steps >= 200:
                    break
            eval_reward.append(total_reward)
        return np.mean(eval_reward)


    def train(self):
        # 往经验池中预存数据
        while self.buffer.size < self.warm_size:
            self.run_episode()


        episode = 0
        self.global_steps = 0  # Reset global steps for fresh training  TODO: review
        while episode < self.train_eposode:
            for i in range(50):
                total_reward = self.run_episode()
                episode += 1

            print("total reward = ", total_reward)

    def process_action(self, action, noise=False):
        """Process action returned from algorithm prediction.

        This method properly process both discrete and continuous action space.
        
        When noise option is enabled:
            For continuous action space, the action is resampled using N(action, noise).
            For discrete action space, the action is chosen by epsilon-greedy.
        
        Additionally, before passing into environment:
            For continuous action space, the action is clipped in interval [-1, 1].
            For discrete action space, the action is passed as is.

        Args:
            action (any): Action to process
            noise (bool, optional): Add noise to action value, useful during training. Defaults to False.

        Returns:
            any: Processed action
        """

        if self.is_conti:
            # Continuous action space
            if noise:
                action = np.random.normal(action, self.noise)
            return np.clip(action, -1.0, 1.0)
        else:
            # Discrete action space
            if noise:
                action = self.epsilon_scheduler.sample(action, self.additional_args['action_num'], self.global_steps)

            return action


class EpsilonScheduler():
    """Epsilon-greedy scheduler with epsilon schedule."""

    def __init__(self, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200):
        """Create an EpsilonScheduler.

        For fixed epsilon-greedy policy, passing EPS_START equal to EPS_END.

        Args:
            EPS_START (float, optional): Epsilon when starting training. Defaults to 0.9.
            EPS_END (float, optional): Epsilon when training infinity steps. Defaults to 0.05.
            EPS_DECAY (float, optional): Exponential coefficient, larger for a slower decay rate (similar to time constant, but for steps). Defaults to 200.
        """
        self.start = EPS_START
        self.end = EPS_END
        self.decay = EPS_DECAY

    def sample(self, action, action_num, steps):
        """Choose an action based on epsilon-greedy policy.

        Args:
            action (any): Predicted action, usually greedy.
            action_num (int): Num of discrete actions.
            steps (int): Global training steps.

        Returns:
            any: Action chosen by psilon-greedy policy.
        """
        thresh = self.end + (self.start - self.end) * np.exp(-steps / self.decay)
        if np.random.random() > thresh:
            return action
        else:
            return np.random.randint(action_num)

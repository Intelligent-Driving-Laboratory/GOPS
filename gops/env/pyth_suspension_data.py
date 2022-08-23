#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Jie Li
#  Description: Suspension Environment
#

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.wrappers.time_limit import TimeLimit
gym.logger.setLevel(gym.logger.ERROR)
import torch

class _GymSuspension(gym.Env):
    def __init__(self, **kwargs):
        """
        you need to define parameters here
        """
        # define common parameters here
        self.is_adversary = kwargs['is_adversary']
        self.state_dim = 4
        self.action_dim = 1
        self.adversary_dim = 1
        self.tau = 1 / 500  # seconds between state updates

        # define your custom parameters here
        self.M_b = 300  # the mass of the car body(kg)
        self.M_us = 60  # the mass of the wheel(kg)
        self.K_t = 190000  # the tyre stiffness(N/m)
        self.K_a = 16000  # the linear suspension stiffness(N/m)
        self.K_n = self.K_a / 10  # the nonlinear suspension stiffness(N/m)
        self.C_a = 1000  # the damping rate of the suspension(N/(m/s))
        self.control_gain = 1e3

        # utility information
        self.Q = np.eye(self.state_dim)
        self.Q[0, 0] = 10000.
        self.Q[1, 1] = 3.
        self.Q[2, 2] = 100.
        self.Q[3, 3] = 0.1
        # self.Q[0, 0] = 1000.
        # self.Q[1, 1] = 10.
        # self.Q[2, 2] = 1000.
        # self.Q[3, 3] = 0.1
        self.R = np.eye(self.action_dim)
        # self.R[0, 0] = 0.2
        self.gamma = 1
        self.gamma_atte = 30.0
        # self.gamma_atte = 0

        # state & action space
        self.pos_body_initial = 0.05
        self.vel_body_initial = 0.5
        self.pos_wheel_initial = 0.05
        self.vel_wheel_initial = 1.0
        self.pos_body_threshold = 0.08
        self.vel_body_threshold = 0.8
        self.pos_wheel_threshold = 0.08
        self.vel_wheel_threshold = 1.6
        self.min_action = [-1.2]
        self.max_action = [1.2]
        self.min_adv_action = [-2.0 / self.gamma_atte]
        self.max_adv_action = [2.0 / self.gamma_atte]

        self.obs_scale = [10, 1, 10, 0.5]
        self.observation_space = spaces.Box(low=np.array([-self.pos_body_threshold, -self.vel_body_threshold,
                                                          -self.pos_wheel_threshold, -self.vel_wheel_threshold]),
                                            high=np.array([self.pos_body_threshold, self.vel_body_threshold,
                                                           self.pos_wheel_threshold, self.vel_wheel_threshold]),
                                            shape=(4,)
                                            )
        # self.action_space = spaces.Box(low=np.array(self.min_action + self.min_adv_action),
        #                                high=np.array(self.max_action + self.max_adv_action),
        #                                shape=(2,)
        #                                )
        self.action_space = spaces.Box(low=np.array(self.min_action),
                                       high=np.array(self.max_action),
                                       shape=(1,)
                                       )

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.max_episode_steps = 1000
        self.steps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, action, adv_action):
        tau = self.tau
        M_b = self.M_b
        M_us = self.M_us
        K_t = self.K_t
        K_a = self.K_a
        K_n = self.K_n
        C_a = self.C_a
        control_gain = self.control_gain
        pos_body, vel_body, pos_wheel, vel_wheel = self.state
        force = action[0]         # the control force of the hydraulic actuator [kN]
        pos_road = adv_action[0]  # the road disturbance

        pos_body_dot = vel_body
        vel_body_dot = - (K_a * (pos_body - pos_wheel) + K_n * pow(pos_body - pos_wheel, 3) +
                          C_a * (vel_body - vel_wheel) - control_gain * force) / M_b
        pos_wheel_dot = vel_wheel
        vel_wheel_dot = (K_a * (pos_body - pos_wheel) + K_n * pow(pos_body - pos_wheel, 3) +
                         C_a * (vel_body - vel_wheel) - K_t * (pos_wheel - pos_road) - control_gain * force) / M_us

        next_pos_body = pos_body_dot * tau + pos_body
        next_vel_body = vel_body_dot * tau + vel_body
        next_pos_wheel = pos_wheel_dot * tau + pos_wheel
        next_vel_wheel = vel_wheel_dot * tau + vel_wheel

        return next_pos_body, next_vel_body, next_pos_wheel, next_vel_wheel

    def step(self, inputs):
        action = inputs[:self.action_dim]
        # adv_action = inputs[self.action_dim:]
        # adv_action = np.random.normal(loc=0, scale=0.0001, size=(1,))
        adv_action = [0]
        if adv_action is None:
            adv_action = 0

        pos_body, vel_body, pos_wheel, vel_wheel = self.state
        self.state = self.stepPhysics(action, adv_action)
        next_pos_body, next_vel_body, next_pos_wheel, next_vel_wheel = self.state
        done = next_pos_body < -self.pos_body_threshold or next_pos_body > self.pos_body_threshold \
            or next_vel_body < -self.vel_body_threshold or next_vel_body > self.vel_body_threshold \
            or next_pos_wheel < -self.pos_wheel_threshold or next_pos_wheel > self.pos_wheel_threshold \
            or next_vel_wheel < -self.vel_wheel_threshold or next_vel_wheel > self.vel_wheel_threshold
        done = bool(done)

        # -----------------
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            done = True
        # ---------------

        if not done:
            # reward = self.Q[0][0] * pos_body ** 2 + self.Q[1][1] * vel_body ** 2 \
            #          + self.Q[2][2] * pos_wheel ** 2 + self.Q[3][3] * vel_wheel ** 2 \
            #          + self.R[0][0] * action ** 2 - self.gamma_atte ** 2 * adv_action ** 2
            reward = self.Q[0][0] * pos_body ** 2 + self.Q[1][1] * vel_body ** 2 \
                     + self.Q[2][2] * pos_wheel ** 2 + self.Q[3][3] * vel_wheel ** 2 \
                     + self.R[0][0] * action ** 2


            # reward = - self.Q[0][0] * pos_body ** 2 - self.Q[1][1] * vel_body ** 2 \
            #          - self.Q[2][2] * (pos_body - pos_wheel) ** 2 - self.R[0][0] * action ** 2
            reward = -reward
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            # reward = self.Q[0][0] * pos_body ** 2 + self.Q[1][1] * vel_body ** 2 \
            #          + self.Q[2][2] * pos_wheel ** 2 + self.Q[3][3] * vel_wheel ** 2 \
            #          + self.R[0][0] * action ** 2 - self.gamma_atte ** 2 * adv_action ** 2

            reward = self.Q[0][0] * pos_body ** 2 + self.Q[1][1] * vel_body ** 2 \
                     + self.Q[2][2] * pos_wheel ** 2 + self.Q[3][3] * vel_wheel ** 2 \
                     + self.R[0][0] * action ** 2
            reward = -reward
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = 0
        state = np.array(self.state)
        obs = self.postprocess(state)
        return obs, reward.item(), done, {}

    def postprocess(self, state):
        obs = np.zeros(self.observation_space.shape)
        obs[0] = state[0]
        obs[1] = state[1]
        obs[2] = state[2]
        obs[3] = state[3]
        obs[0:4] = obs[0:4]*self.obs_scale


        return obs

    def reset(self):
        self.state = self.np_random.uniform(low=[-self.pos_body_initial, -self.vel_body_initial,
                                                 -self.pos_wheel_initial, -self.vel_wheel_initial],
                                            high=[self.pos_body_initial, self.vel_body_initial,
                                                  self.pos_wheel_initial, self.vel_wheel_initial],
                                            size=(4,)
                                            )
        self.steps_beyond_done = None
        self.steps = 0
        state = np.array(self.state)
        obs = self.postprocess(state)
        return obs

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()


def env_creator(**kwargs):
    return TimeLimit(_GymSuspension(**kwargs), 1000)


if __name__ == '__main__':
    # obs = np.array([1, 2, 3, 4])
    # obs_1 = obs * [10, 1, 10, 0.5]
    # print(obs_1)
    pass

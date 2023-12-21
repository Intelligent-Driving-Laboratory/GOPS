#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Oscillator Model
#  Update Date: 2022-08-12, Jie Li: create environment
#  Update Date: 2022-10-24, Yujie Yang: add wrapper

from math import sin, cos, exp

import gym
import numpy as np
from gym import spaces
from gops.env.env_ocp.pyth_base_env import PythBaseEnv

gym.logger.setLevel(gym.logger.ERROR)


class _GymOscillatorconti(PythBaseEnv):
    def __init__(self, **kwargs):
        """
        you need to define parameters here
        """
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [battery_a, battery_b]
            init_high = np.array([1.5, 1.5], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(_GymOscillatorconti, self).__init__(work_space=work_space, **kwargs)

        # define common parameters here
        self.is_adversary = kwargs["is_adversary"]
        self.state_dim = 2
        self.action_dim = 1
        self.adversary_dim = 1
        self.tau = 1 / 200
        self.prob_intensity = (
            kwargs["prob_intensity"]
            if kwargs.get("prob_intensity") is not None
            else 1.0
        )
        self.base_decline = (
            kwargs["base_decline"] if kwargs.get("base_decline") is not None else 0.0
        )

        # utility information
        self.Q = np.eye(self.state_dim)
        self.R = np.eye(self.action_dim)
        self.gamma = 1
        self.gamma_atte = kwargs["gamma_atte"]

        # state & action space
        self.state_threshold = kwargs["state_threshold"]
        self.battery_a_threshold = self.state_threshold[0]
        self.battery_b_threshold = self.state_threshold[1]
        self.max_action = [5.0]
        self.min_action = [-5.0]
        self.max_adv_action = [1.0 / self.gamma_atte]
        self.min_adv_action = [-1.0 / self.gamma_atte]

        self.observation_space = spaces.Box(
            low=np.array([-self.battery_a_threshold, -self.battery_b_threshold]),
            high=np.array([self.battery_a_threshold, self.battery_b_threshold]),
            shape=(2,),
        )
        self.action_space = spaces.Box(
            low=np.array(self.min_action), high=np.array(self.max_action), shape=(1,)
        )

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.max_episode_steps = 200
        self.steps = 0

    @property
    def has_optimal_controller(self):
        return True

    def control_policy(self, obs, info):
        return np.array([-obs[0] * obs[1]], dtype="f")

    def reset(self, init_state=None, **kwargs):
        if init_state is None:
            self.state = self.sample_initial_state()
        else:
            self.state = np.array(init_state, dtype=np.float32)
        self.steps_beyond_done = None
        self.steps = 0
        return self.state

    def stepPhysics(self, action, adv_action):

        tau = self.tau
        battery_a, battery_b = self.state
        # memritor
        memristor = action[0]
        # noise
        noise = adv_action[0]

        battery_a_dot = -0.25 * battery_a
        battery_b_dot = (
            0.5 * battery_a ** 2 * battery_b
            - 1 / (2 * self.gamma_atte ** 2) * battery_b ** 3
            - 0.5 * battery_b
            + battery_a * memristor
            + battery_b * noise
        )

        next_battery_a = battery_a_dot * tau + battery_a
        next_battery_b = battery_b_dot * tau + battery_b
        return next_battery_a, next_battery_b

    def step(self, inputs):
        action = inputs[: self.action_dim]
        adv_action = inputs[self.action_dim :]
        if not adv_action or adv_action is None:
            adv_action = [0]

        battery_a, battery_b = self.state
        self.state = self.stepPhysics(action, adv_action)
        next_battery_a, next_battery_b = self.state
        done = (
            next_battery_a < -self.battery_a_threshold
            or next_battery_a > self.battery_a_threshold
            or next_battery_b < -self.battery_b_threshold
            or next_battery_b > self.battery_b_threshold
        )
        done = bool(done)

        # -----------------
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            done = True
        # ---------------

        if not done:
            reward = (
                self.Q[0][0] * battery_a ** 2
                + self.Q[1][1] * battery_b ** 2
                + self.R[0][0] * action[0] ** 2
                - self.gamma_atte ** 2 * adv_action[0] ** 2
            )
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = (
                self.Q[0][0] * battery_a ** 2
                + self.Q[1][1] * battery_b ** 2
                + self.R[0][0] * action[0] ** 2
                - self.gamma_atte ** 2 * adv_action[0] ** 2
            )
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn(
                    """
                You are calling 'step()' even though this environment has already returned
                done = True. You should always call 'reset()' once you receive 'done = True'
                Any further steps are undefined behavior.
                """
                )
            self.steps_beyond_done += 1
            reward = 0.0

        reward_positive = (
            self.Q[0][0] * battery_a ** 2
            + self.Q[1][1] * battery_b ** 2
            + self.R[0][0] * action[0] ** 2
        )
        reward_negative = adv_action[0] ** 2

        return (
            np.array(self.state),
            reward,
            done,
            {"reward_positive": reward_positive, "reward_negative": reward_negative},
        )

    def exploration_noise(self, time):
        n = (
            sin(time) ** 2 * cos(time)
            + sin(2 * time) ** 2 * cos(0.1 * time)
            + sin(1.2 * time) ** 2 * cos(0.5 * time)
            + sin(time) ** 5
            + sin(1.12 * time) ** 2
            + sin(2.4 * time) ** 3 * cos(2.4 * time)
        )
        return np.array([self.prob_intensity * exp(self.base_decline * time) * n, 0])

    @staticmethod
    def init_obs():
        return np.array([0.4, 0.5], dtype="float32")

    @staticmethod
    def dist_func(time):
        t0 = 3.0  # 3.0
        dist = (
            [3.0 * exp(-1.0 * (time - t0)) * cos(1.0 * (time - t0))]
            if time > t0
            else [0.0]
        )
        return dist

    def render(self, mode="human"):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()


def env_creator(**kwargs):
    return _GymOscillatorconti(**kwargs)

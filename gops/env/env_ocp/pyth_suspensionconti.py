#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Suspension Model
#  Update Date: 2022-08-12, Jie Li: create environment
#  Update Date: 2022-10-24, Yujie Yang: add wrapper

from math import sin, cos, sqrt, exp, pi

import gym
import numpy as np
from gym import spaces
from gops.env.env_ocp.pyth_base_env import PythBaseEnv

gym.logger.setLevel(gym.logger.ERROR)


class _GymSuspensionconti(PythBaseEnv):
    def __init__(self, **kwargs):
        """
        you need to define parameters here
        """
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [pos_body, vel_body, pos_wheel, vel_wheel]
            init_high = np.array([0.05, 0.5, 0.05, 1.0], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(_GymSuspensionconti, self).__init__(work_space=work_space, **kwargs)

        # define common parameters here
        self.is_adversary = kwargs["is_adversary"]
        self.state_dim = 4
        self.action_dim = 1
        self.adversary_dim = 1
        self.tau = 1 / 500
        self.prob_intensity = kwargs.get("prob_intensity", 1.0)
        self.base_decline = kwargs.get("base_decline", 0.0)
        self.start_decline = kwargs.get("start_decline", 0)
        self.start_cancel = kwargs.get("start_cancel", kwargs["max_iteration"])
        self.dist_func_type = kwargs.get("dist_func_type", "zero")
        self.initial_obs = kwargs.get("initial_obs", kwargs["fixed_initial_state"])
        self.sample_batch_size = kwargs["reset_batch_size"]

        self.time_start_decline = self.start_decline * self.tau * self.sample_batch_size
        self.time_start_cancel = self.start_cancel * self.tau * self.sample_batch_size

        # define your custom parameters here
        # the mass of the car body [kg]
        self.M_b = 300
        # the mass of the wheel [kg]
        self.M_us = 60
        # the tyre stiffness [N/m]
        self.K_t = 190000
        # the linear suspension stiffness [N/m]
        self.K_a = 16000
        # the nonlinear suspension stiffness [N/m]
        self.K_n = self.K_a / 10
        # the damping rate of the suspension [N / (m/s)]
        self.C_a = 1000
        self.control_gain = 1e3

        # utility information
        self.state_weight = kwargs["state_weight"]
        self.control_weight = kwargs["control_weight"]
        self.Q = np.zeros((self.state_dim, self.state_dim))
        self.Q[0][0] = self.state_weight[0]
        self.Q[1][1] = self.state_weight[1]
        self.Q[2][2] = self.state_weight[2]
        self.Q[3][3] = self.state_weight[3]
        self.R = np.zeros((self.action_dim, self.action_dim))
        self.R[0][0] = self.control_weight[0]
        self.gamma = 1
        self.gamma_atte = kwargs["gamma_atte"]

        # state & action space
        self.state_threshold = kwargs["state_threshold"]
        self.pos_body_threshold = self.state_threshold[0]
        self.vel_body_threshold = self.state_threshold[1]
        self.pos_wheel_threshold = self.state_threshold[2]
        self.vel_wheel_threshold = self.state_threshold[3] * 5
        self.min_action = [-1.2]
        self.max_action = [1.2]
        self.min_adv_action = [-2.0 / self.gamma_atte]
        self.max_adv_action = [2.0 / self.gamma_atte]

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -self.pos_body_threshold,
                    -self.vel_body_threshold,
                    -self.pos_wheel_threshold,
                    -self.vel_wheel_threshold,
                ]
            ),
            high=np.array(
                [
                    self.pos_body_threshold,
                    self.vel_body_threshold,
                    self.pos_wheel_threshold,
                    self.vel_wheel_threshold,
                ]
            ),
            shape=(self.state_dim,),
        )
        self.action_space = spaces.Box(
            low=np.array(self.min_action),
            high=np.array(self.max_action),
            shape=(self.action_dim,),
        )

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

        self.max_episode_steps = 1500
        self.steps = 0

    def reset(self, init_state=None, **kwargs):
        if init_state is None:
            self.state = self.sample_initial_state()
        else:
            self.state = np.array(init_state, dtype=np.float32)
        self.steps_beyond_done = None
        self.steps = 0
        return self.state

    def reload_para(self):
        pass

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
        # the control force of the hydraulic actuator [kN]
        force = action[0]
        # the road disturbance [m]
        pos_road = adv_action[0]

        pos_body_dot = vel_body
        vel_body_dot = (
            -(
                K_a * (pos_body - pos_wheel)
                + K_n * pow(pos_body - pos_wheel, 3)
                + C_a * (vel_body - vel_wheel)
                - control_gain * force
            )
            / M_b
        )
        pos_wheel_dot = vel_wheel
        vel_wheel_dot = (
            K_a * (pos_body - pos_wheel)
            + K_n * pow(pos_body - pos_wheel, 3)
            + C_a * (vel_body - vel_wheel)
            - K_t * (pos_wheel - pos_road)
            - control_gain * force
        ) / M_us

        next_pos_body = pos_body_dot * tau + pos_body
        next_vel_body = vel_body_dot * tau + vel_body
        next_pos_wheel = pos_wheel_dot * tau + pos_wheel
        next_vel_wheel = vel_wheel_dot * tau + vel_wheel
        return next_pos_body, next_vel_body, next_pos_wheel, next_vel_wheel

    def step(self, inputs):
        action = inputs[: self.action_dim]
        adv_action = inputs[self.action_dim :]
        if not adv_action or adv_action is None:
            adv_action = [0]

        pos_body, vel_body, pos_wheel, vel_wheel = self.state
        self.state = self.stepPhysics(action, adv_action)
        next_pos_body, next_vel_body, next_pos_wheel, next_vel_wheel = self.state
        done = (
            next_pos_body < -self.pos_body_threshold
            or next_pos_body > self.pos_body_threshold
            or next_vel_body < -self.vel_body_threshold
            or next_vel_body > self.vel_body_threshold
            or next_pos_wheel < -self.pos_wheel_threshold
            or next_pos_wheel > self.pos_wheel_threshold
            or next_vel_wheel < -self.vel_wheel_threshold
            or next_vel_wheel > self.vel_wheel_threshold
        )
        done = bool(done)

        # -----------------
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            done = True
        # ---------------

        if not done:
            reward = (
                self.Q[0][0] * pos_body ** 2
                + self.Q[1][1] * vel_body ** 2
                + self.Q[2][2] * pos_wheel ** 2
                + self.Q[3][3] * vel_wheel ** 2
                + self.R[0][0] * action[0] ** 2
                - self.gamma_atte ** 2 * adv_action[0] ** 2
            )
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = (
                self.Q[0][0] * pos_body ** 2
                + self.Q[1][1] * vel_body ** 2
                + self.Q[2][2] * pos_wheel ** 2
                + self.Q[3][3] * vel_wheel ** 2
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
            self.Q[0][0] * pos_body ** 2
            + self.Q[1][1] * vel_body ** 2
            + self.Q[2][2] * pos_wheel ** 2
            + self.Q[3][3] * vel_wheel ** 2
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
        if time > self.time_start_cancel:
            self.prob_intensity = 0
        if time < self.time_start_decline:
            final_prob_intensity = self.prob_intensity
        else:
            final_prob_intensity = self.prob_intensity * exp(
                self.base_decline * (time - self.time_start_decline)
            )
        return np.array([final_prob_intensity * n, 0])

    @staticmethod
    def init_obs():
        return np.array([0, 0, 0, 0], dtype="float32")

    @staticmethod
    def dist_func(time):
        dist = [0.038 * (1 - cos(8 * pi * time))] if 0.5 <= time < 0.75 else [0.0]
        return np.array(dist)

    def render(self, mode="human"):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()


def dist_func_sine_noise(time):
    t0 = 0.0
    dist = [0.5 * sin(2 * sqrt(2 / 3) * (time - t0))]
    return dist


def env_creator(**kwargs):
    return _GymSuspensionconti(**kwargs)

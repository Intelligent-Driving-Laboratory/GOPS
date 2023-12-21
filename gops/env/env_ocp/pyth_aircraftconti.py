#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Aircraft Model
#  Update Date: 2022-08-12, Jie Li: create environment
#  Update Date: 2022-10-24, Yujie Yang: add wrapper

from math import sin, cos

import gym
import numpy as np
from gym import spaces
from gops.env.env_ocp.pyth_base_env import PythBaseEnv

gym.logger.setLevel(gym.logger.ERROR)


class _GymAircraftconti(PythBaseEnv):
    def __init__(self, **kwargs):
        """
        you need to define parameters here
        """
        work_space = kwargs.pop("work_space", None)
        if work_space is None:
            # initial range of [attack_ang, rate, elevator_ang]
            init_high = np.array([0.3, 0.6, 0.3], dtype=np.float32)
            init_low = -init_high
            work_space = np.stack((init_low, init_high))
        super(_GymAircraftconti, self).__init__(work_space=work_space, **kwargs)

        # define common parameters here
        self.is_adversary = kwargs["is_adversary"]
        self.state_dim = 3
        self.action_dim = 1
        self.adversary_dim = 1
        self.tau = 1 / 200  # seconds between state updates

        # define your custom parameters here
        self.A = np.array(
            [[-1.01887, 0.90506, -0.00215], [0.82225, -1.07741, -0.17555], [0, 0, -1]],
            dtype=np.float32,
        )
        self.A_attack_ang = np.array(
            [-1.01887, 0.90506, -0.00215], dtype=np.float32
        ).reshape((3, 1))
        self.A_rate = np.array([0.82225, -1.07741, -0.17555], dtype=np.float32).reshape(
            (3, 1)
        )
        self.A_elevator_ang = np.array([0, 0, -1], dtype=np.float32).reshape((3, 1))
        self.B = np.array([0.0, 0.0, 1.0]).reshape((3, 1))
        self.D = np.array([1.0, 0.0, 0.0]).reshape((3, 1))

        # utility information
        self.Q = np.eye(self.state_dim)
        self.R = np.eye(self.action_dim)
        self.gamma = 1
        self.gamma_atte = kwargs["gamma_atte"]
        self.control_matrix = np.array(
            [[0.166065, 0.180362, -0.437060]], dtype=np.float32
        )

        # state & action space
        self.state_threshold = kwargs["state_threshold"]
        self.attack_ang_threshold = self.state_threshold[0]
        self.rate_threshold = self.state_threshold[1]
        self.elevator_ang_threshold = self.state_threshold[2]
        self.max_action = [1.0]
        self.min_action = [-1.0]
        self.max_adv_action = [1.0 / self.gamma_atte]
        self.min_adv_action = [-1.0 / self.gamma_atte]

        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -self.attack_ang_threshold,
                    -self.rate_threshold,
                    -self.elevator_ang_threshold,
                ]
            ),
            high=np.array(
                [
                    self.attack_ang_threshold,
                    self.rate_threshold,
                    self.elevator_ang_threshold,
                ]
            ),
            shape=(3,),
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
        action = self.control_matrix @ obs
        return action

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
        A = self.A
        attack_ang, rate, elevator_ang = self.state
        # the elevator actuator voltage
        elevator_vol = action[0]
        # wind gusts on angle of attack
        wind_attack_angle = adv_action[0]

        attack_ang_dot = (
            A[0, 0] * attack_ang
            + A[0, 1] * rate
            + A[0, 2] * elevator_ang
            + wind_attack_angle
        )
        rate_dot = A[1, 0] * attack_ang + A[1, 1] * rate + A[1, 2] * elevator_ang
        elevator_ang_dot = (
            A[2, 0] * attack_ang
            + A[2, 1] * rate
            + A[2, 2] * elevator_ang
            + elevator_vol
        )

        next_attack_ang = attack_ang_dot * tau + attack_ang
        next_rate = rate_dot * tau + rate
        next_elevator_angle = elevator_ang_dot * tau + elevator_ang
        return next_attack_ang, next_rate, next_elevator_angle

    def step(self, inputs):
        action = inputs[: self.action_dim]
        adv_action = inputs[self.action_dim :]
        if not adv_action or adv_action is None:
            adv_action = [0]

        attack_ang, rate, elevator_ang = self.state
        self.state = self.stepPhysics(action, adv_action)
        next_attack_ang, next_rate, next_elevator_angle = self.state
        done = (
            next_attack_ang < -self.attack_ang_threshold
            or next_attack_ang > self.attack_ang_threshold
            or next_rate < -self.rate_threshold
            or next_rate > self.rate_threshold
            or next_elevator_angle < -self.elevator_ang_threshold
            or next_elevator_angle > self.elevator_ang_threshold
        )
        done = bool(done)

        # -----------------
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            done = True
        # ---------------

        if not done:
            reward = (
                self.Q[0][0] * attack_ang ** 2
                + self.Q[1][1] * rate ** 2
                + self.Q[2][2] * elevator_ang ** 2
                + self.R[0][0] * action[0] ** 2
                - self.gamma_atte ** 2 * adv_action[0] ** 2
            )
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = (
                self.Q[0][0] * attack_ang ** 2
                + self.Q[1][1] * rate ** 2
                + self.Q[2][2] * elevator_ang ** 2
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

        return np.array(self.state), reward, done, {}

    @staticmethod
    def exploration_noise(time):
        n = (
            sin(time) ** 2 * cos(time)
            + sin(2 * time) ** 2 * cos(0.1 * time)
            + sin(1.2 * time) ** 2 * cos(0.5 * time)
            + sin(time) ** 5
            + sin(1.12 * time) ** 2
            + sin(2.4 * time) ** 3 * cos(2.4 * time)
        )
        return np.array([n, 0])

    def render(self, mode="human"):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()


def env_creator(**kwargs):
    return _GymAircraftconti(**kwargs)

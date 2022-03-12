#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Tracking Car Environment
#  Update Date: 2021-05-55, Yuhang Zhang: create environment



import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.wrappers.time_limit import TimeLimit

class _GymTrackingCar(gym.Env):
    def __init__(self,**kwargs):
        self.a = 1.463
        self.b = 1.585
        self.m = 1818.2
        self.Iz = 3885
        self.kf = -62618
        self.kr = -110185

        self.speed = 10  # constant speed
        self.gravity = 9.8

        self.control_tau = 0.1  # seconds between control commands
        self.dynamic_tau = 0.001  # seconds between state updates
        self.step_count = int(self.control_tau / self.dynamic_tau)

        self.max_action = 0.1  # max front wheel angle: 0.1 rad , ay = 2.7 m/s^2
        self.min_action = -self.max_action

        # Angle at which to fail the episode
        self.theta_threshold = math.pi / 4
        self.y_threshold = 1.75  # m
        self.ay_threshold = 3  # m/s^2
        self.fell_reward = -100

        self.rw_delta = 10
        self.rw_y = 1

        # create action space and observation_space
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,)
        )

        high = np.array([
            self.y_threshold,
            self.theta_threshold])
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.state = None
        self.vehicle_state = None
        self.steps_beyond_done = None  # number of step after done flag

        self._max_episode_steps = kwargs.get('max_episode_steps',50)
        self.steps = 0
        self.fell = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        reset_y = 1.5
        self.state = np.zeros((2,), dtype=np.float32)
        self.state[0] = reset_y
        self.steps_beyond_done = None
        self.steps = 0
        # -------------
        self.vehicle_state = np.zeros((6,), dtype=np.float32)
        self.vehicle_state[1] = self.state[0]  # y
        self.vehicle_state[2] = self.speed  # speed
        # -------------
        return np.array(self.state)

    # def reset_target(self, target_y=-1.5):
    #     reset_y = target_y
    #     self.state = np.zeros((2,), dtype=np.float32)
    #     self.state[0] = reset_y
    #     self.steps_beyond_done = None
    #     self.steps = 0
    #     # -------------
    #     self.vehicle_state = np.zeros((6,), dtype=np.float32)
    #     self.vehicle_state[1] = self.state[0]  # y
    #     self.vehicle_state[2] = self.speed  # speed
    #     # -------------
    #     return np.array(self.state)

    def stepPhysics(self, delta):
        # step 0.1s
        cmd = np.zeros((2,), dtype=np.float32)
        cmd[1] = delta
        for _ in range(self.step_count):
            self.vehicle_state = self._update_data(x0=self.vehicle_state, u0=cmd, T=self.dynamic_tau)
        return self.vehicle_state

    def get_vehicle_state(self):
        return self.vehicle_state

    def step(self, action):
        action = np.expand_dims(action, 0)
        action = action.clip(self.min_action, self.max_action)
        delta = float(action)
        veh_state = self.stepPhysics(delta)
        self.state[0] = veh_state[1]
        self.state[1] = veh_state[4]
        y = veh_state[1]
        theta = veh_state[4]

        self.steps += 1
        # if self.steps >= self._max_episode_steps:
        #     done = True

        done = False
        fell = False
        if y > self.y_threshold \
                or y < -self.y_threshold \
                or theta > self.theta_threshold \
                or theta < -self.theta_threshold:
            fell = True
        self.fell = fell
        if fell:
            done = True

        # ---------------
        if not done:
            reward = self._get_reward(delta=delta, y=y)
        elif self.steps_beyond_done is None:
            # Car just fell! Normal
            self.steps_beyond_done = 0
            reward = self._get_reward(delta=delta, y=y)
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn("""
    You are calling 'step()' even though this environment has already returned
    done = True. You should always call 'reset()' once you receive 'done = True'
    Any further steps are undefined behavior.
                    """)
            self.steps_beyond_done += 1

            reward = self._get_reward(delta=delta, y=y)

        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        print("Tracking car do not have render function!")

    def _get_reward(self, delta, y):

        if np.fabs(y) > self.y_threshold:
            y = self.y_threshold

        if np.fabs(delta) > self.max_action:
            delta = self.max_action

        r_y = y / self.y_threshold
        r_d = delta / self.max_action

        alpha_max = 20
        alpha_min = 10

        kp = np.fabs(y) / self.y_threshold
        alpha = alpha_max + kp * (alpha_min - alpha_max)

        reward_y_err = - alpha * r_y * r_y
        reward_delta = -  r_d * r_d

        reward_fell = 0
        if self.fell:
            reward_fell = -100

        return (reward_y_err + reward_delta + reward_fell)

    def _update_data(self, x0, u0, T):
        x1 = np.zeros(len(x0))
        x1[0] = x0[0] + T * (x0[2] * np.cos(x0[4]) - x0[3] * np.sin(x0[4]))
        x1[1] = x0[1] + T * (x0[3] * np.cos(x0[4]) + x0[2] * np.sin(x0[4]))
        x1[2] = x0[2] + T * u0[0]  # vel.x
        x1[3] = (-(self.a * self.kf - self.b * self.kr) * x0[5] +
                 self.kf * u0[1] * x0[2] + self.m * x0[5] * x0[2] * x0[2] -
                 self.m * x0[2] * x0[3] / T) / (self.kf + self.kr - self.m * x0[2] / T)
        x1[4] = x0[4] + T * x0[5]
        x1[5] = (-self.Iz * x0[5] * x0[2] / T - (self.a * self.kf - self.b * self.kr) * x0[3] +
                 self.a * self.kf * u0[1] * x0[2]) / (
                        (self.a * self.a * self.kf + self.b * self.b * self.kr) - self.Iz * x0[2] / T)
        return x1

def env_creator(**kwargs):
    _max_episode_steps = kwargs.get('max_episode_steps', 50)
    return TimeLimit(_GymTrackingCar(**kwargs), _max_episode_steps)


if __name__=="__main__":
    env = env_creator()
    print(env.action_space.high)
    print(env.action_space.low)
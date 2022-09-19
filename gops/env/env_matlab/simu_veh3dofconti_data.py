#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Vehicle 3DOF Model
#  Update Date: 2021-05-55, Congsheng Zhang: create environment


from gym import spaces
import gym
from gym.utils import seeding
from gops.env.env_matlab.resources.simu_vehicle3dof_v2 import vehicle3dof
from gym.wrappers.time_limit import TimeLimit
from typing import List
import numpy as np
import copy

Max_Step_default = 500
ref_T_default = [30, 60, 180]
ref_A_default = [0.3, 0.8, 1.5]
ref_fai_default = [0, np.pi / 6, np.pi / 3]
ref_V_default = 20

Q_default = [0.04, 0.01, 0.1, 0.02]  # v_x
R_default = [4, 1e-8, 1e-8]
rew_bound_default = 20000
rew_bias_default = 0
punish_done_default = -2000
action_scale_default = [1, 1. / 1000, 1. / 1000]
a_max_default = [1.5, 3000, 3000]
obs_scale_default = [1 / 1200, 1, 1, 1, 2.4, 2]
random_bias_default = [50., 2.5, 1, 1, np.pi / 9, 0.3]
random_method_default = 'uniform'
done_default = [3, 5, np.pi/4]
ref_horizon_default = 0


class Ref_Curve:
    def __init__(self,
                 ref_a: List,
                 ref_t: List,
                 ref_fai: List,
                 ref_v,
                 ):
        self.A = ref_a
        self.T = ref_t
        self.fai = ref_fai
        self.V = ref_v

    def cal_reference(self, pos_x):
        pos_y = 0
        k_y = 0
        for items in zip(self.A, self.T, self.fai):
            pos_y += items[0] * np.sin(2 * np.pi / items[1] * pos_x + items[2])
            k_y += 2 * np.pi / items[1] * items[0] * np.cos(2 * np.pi / items[1] * pos_x + items[2])
        return pos_y, np.arctan(k_y), self.V


class SimuVeh3dofconti(gym.Env):
    def __init__(self, **kwargs):
        self._physics = vehicle3dof.model_wrapper()
        self.is_adversary = kwargs.get("is_adversary", False)
        self.Max_step = kwargs.get("Max_step", Max_Step_default)

        self.act_scale = np.array(kwargs.get('act_scale', action_scale_default))
        self.a_max = np.array(kwargs.get('a_max',a_max_default))
        self.a_min = - self.a_max
        self.obs_scale = np.array(kwargs.get('obs_scale', obs_scale_default))
        self.rew_bound = kwargs.get('rew_bound', rew_bound_default)
        self.rew_bias = kwargs.get('rew_bias', rew_bias_default)
        self.punish_done = kwargs.get('punish_done', punish_done_default)
        self.done_range = kwargs.get('done_range', done_default)
        self.ref_horizon = kwargs.get('ref_horizon', ref_horizon_default)
        ref_A = kwargs.get('ref_A', ref_A_default)
        ref_T = kwargs.get('ref_T', ref_T_default)
        ref_fai = kwargs.get('ref_fai', ref_fai_default)
        ref_V = kwargs.get('ref_V', ref_V_default)
        self.ref_curve = Ref_Curve(ref_A, ref_T, ref_fai, ref_V)

        self.Q = kwargs.get('punish_Q', Q_default)
        self.R = kwargs.get('punish_R', R_default)

        self.random_base = np.array([0., ref_V, 0., 0., 0., 0.])
        self.random_bias = np.array(kwargs.get('rand_bias', random_bias_default))

        self.rand_low = self.random_base - self.random_bias
        self.rand_high = self.random_base + self.random_bias
        self.rand_method = kwargs.get('rand_method', random_method_default)

        self.action_space = spaces.Box(
            self.act_scale * self.a_min,
            self.act_scale * self.a_max,
        )
        obs_low = self.obs_scale * np.array(self._physics.get_param()["x_min"]).reshape(-1)
        ref_pos_low = -self.obs_scale[1]*self.done_range[0]*np.zeros(self.ref_horizon)
        ref_pos_high = self.obs_scale[1] * self.done_range[0] * np.zeros(self.ref_horizon)

        obs_low = np.concatenate([obs_low, ref_pos_low])
        obs_high = self.obs_scale * np.array(self._physics.get_param()["x_max"]).reshape(-1)
        obs_high = np.concatenate([obs_high, ref_pos_high])
        self.observation_space = spaces.Box(obs_low[1:], obs_high[1:])

        self.adv_action_space = spaces.Box(
            self.act_scale * np.array(self._physics.get_param()["adva_min"]).reshape(-1),
            self.act_scale * np.array(self._physics.get_param()["adva_max"]).reshape(-1),
        )
        self.adv_action_dim = self.adv_action_space.shape[0]
        self.seed()
        self._state = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def step(self, action, adv_action=None):
        action = action / self.act_scale
        if not self.is_adversary:
            if adv_action is not None:
                raise ValueError("Adversary training setting is wrong")
            else:
                adv_action = np.array([0.0] * self.adv_action_dim)
        else:
            if adv_action is None:
                raise ValueError("Adversary training setting is wrong")
        state, isdone, reward = self._step_physics(
            {"Action": action.astype(np.float64), "AdverAction": adv_action.astype(np.float64)}
        )
        ref_y, ref_phi, ref_v = self.ref_curve.cal_reference(state[0])

        isdone = abs(state[1] - ref_y) > self.done_range[0]\
                 or abs(state[2]-ref_v) > self.done_range[1] \
                 or abs(state[4]-ref_phi) > self.done_range[2]
        Q = self.Q
        R = self.R
        reward = Q[0] * (state[1] - ref_y) ** 2 + \
                 Q[1] * (state[2] - ref_v) ** 2 + \
                 Q[2] * (state[4] - ref_phi) ** 2 + \
                 Q[3] * (state[5]) ** 2 + \
                 R[0] * action[0] ** 2 + \
                 R[1] * action[1] ** 2 + \
                 R[2] * action[2] ** 2

        reward = -reward
        if reward < -self.rew_bound:
            reward = -self.rew_bound

        reward = reward + self.rew_bias
        if isdone:
            reward = reward + self.punish_done
        self.cstep += 1
        info = {"TimeLimit.truncated": self.cstep > self.Max_step}
        state[1] = ref_y - state[1]
        state[4] = ref_phi - state[4]
        x_pre = state[0] + ref_v*0.02*np.linspace(1, self.ref_horizon,self.ref_horizon)
        y_pre, _, _ = self.ref_curve.cal_reference(x_pre)
        y_pre = (y_pre - state[1])*self.obs_scale[1]
        state = state * self.obs_scale
        self._state = np.concatenate([state, y_pre])


        return self._state[1:], reward, isdone, info

    def reset(self, state = None):
        self._physics.terminate()
        self._physics = vehicle3dof.model_wrapper()
        if state is None:
        # randomized initiate
            if self.rand_method == 'gauss':
                state = self.random_base + self.random_bias*np.random.randn(self.rand_low.size)
            elif self.rand_method == 'uniform':
                state = np.random.uniform(low=self.rand_low, high=self.rand_high)
            else:
                raise NotImplemented

        ref_y, ref_phi, ref_v = self.ref_curve.cal_reference(state[0])


        param = self._physics.get_param()
        param.update({"x_ini": state})
        self._physics.set_param(param)
        self._physics.initialize()

        state[2] = -state[2]
        state[4] = -state[4]
        self.cstep = 0
        state_r = copy.copy(state)
        state_r[1] = state[2]
        state_r[2] = state[1]
        self._state = state_r * self.obs_scale
        return self._state[1:]

    @property
    def state(self):
        return self._state

    def render(self):
        pass

    def close(self):
        # self._physics.renderterminate()
        pass

    def _step_physics(self, action):
        return self._physics.step(action)


def env_creator(**kwargs):
    """
    make env `simu_veh3dofconti` from
    """
    return TimeLimit(SimuVeh3dofconti(**kwargs), kwargs.get("Max_step", Max_Step_default))


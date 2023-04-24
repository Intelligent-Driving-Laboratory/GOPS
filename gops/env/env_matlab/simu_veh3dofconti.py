#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Simulink Vehicle 3Dof data environment
#  Update Date: 2021-07-011, Wenxuan Wang: create simulink environment

from typing import Optional, List, Tuple, Any, Sequence

from gym import spaces
import gym
from gym.utils import seeding
import numpy as np
from gops.env.env_matlab.resources.simu_vehicle3dof_v2 import vehicle3dof


class RefCurve:
    def __init__(
        self, ref_a: List, ref_t: List, ref_fai: List, ref_v: float,
    ):
        self.A = ref_a
        self.T = ref_t
        self.fai = ref_fai
        self.V = ref_v

    def cal_reference(self, pos_x: float) -> Tuple[float, float, float]:
        pos_y = 0
        k_y = 0
        for items in zip(self.A, self.T, self.fai):
            pos_y += items[0] * np.sin(2 * np.pi / items[1] * pos_x + items[2])
            k_y += (
                2
                * np.pi
                / items[1]
                * items[0]
                * np.cos(2 * np.pi / items[1] * pos_x + items[2])
            )
        return pos_y, np.arctan(k_y), self.V


class SimuVeh3dofconti(gym.Env,):
    def __init__(self, **kwargs: Any):
        spec = vehicle3dof._env.EnvSpec(
            id="SimuVeh3dofConti-v0",
            max_episode_steps=kwargs["Max_step"],
            terminal_bonus_reward=kwargs["punish_done"],
            strict_reset=True,
        )
        self.env = vehicle3dof.GymEnv(spec)
        self.dt = 0.01
        self.is_adversary = kwargs.get("is_adversary", False)
        self.obs_scale = np.array(kwargs["obs_scaling"])
        self.act_scale = np.array(kwargs["act_scaling"])
        self.act_max = np.array(kwargs["act_max"])
        self.done_range = kwargs["done_range"]
        self.punish_done = kwargs["punish_done"]
        self.use_ref = kwargs["ref_info"]
        self.ref_horizon = kwargs["ref_horizon"]
        self._state = None

        obs_low = self.obs_scale * np.array([-9999, -9999, -9999, -9999, -9999, -9999])
        ref_pos_low = (
            -self.obs_scale[1] * self.done_range[0] * np.ones(self.ref_horizon)
        )
        ref_phi_low = (
            -self.obs_scale[4] * self.done_range[2] * np.ones(self.ref_horizon)
        )
        if self.use_ref == "None":
            self.observation_space = spaces.Box(obs_low, -obs_low)
        elif self.use_ref == "Pos":
            obs_low = np.concatenate([obs_low, ref_pos_low])
            self.observation_space = spaces.Box(obs_low, -obs_low)
        elif self.use_ref == "Both":
            obs_low = np.concatenate([obs_low, ref_pos_low])
            obs_low = np.concatenate([obs_low, ref_phi_low])
            self.observation_space = spaces.Box(obs_low, -obs_low)
        else:
            raise NotImplementedError

        self.action_space = self.env.action_space
        self.action_space = spaces.Box(
            -self.act_scale * self.act_max, self.act_scale * self.act_max,
        )
        # Split RNG, if randomness is needed
        self.rng = np.random.default_rng()

        self.reward_bias = kwargs["rew_bias"]
        self.reward_bound = kwargs["rew_bound"]
        self.act_repeat = kwargs["act_repeat"]
        self.rand_bias = kwargs["rand_bias"]
        self.rand_center = kwargs["rand_center"]
        ref_A = kwargs["ref_A"]
        ref_T = kwargs["ref_T"]
        ref_fai = kwargs["ref_fai"]
        ref_V = kwargs["ref_V"]

        self.Q = np.array(kwargs["punish_Q"])
        self.R = np.array(kwargs["punish_R"])
        self.ref_curve = RefCurve(ref_A, ref_T, ref_fai, ref_V)

        self.rand_low = np.array(self.rand_center) - np.array(self.rand_bias)
        self.rand_high = np.array(self.rand_center) + np.array(self.rand_bias)
        self.seed()
        self.reset()

    @property
    def state(self):
        return self._state

    def reset(
        self, init_state: Optional[Sequence] = None, **kwargs: Any
    ) -> Tuple[np.ndarray]:
        def callback():
            """Custom reset logic goes here."""
            # Modify your parameter
            # e.g. self.env.model_class.foo_InstP.your_parameter
            if init_state is None:
                self._state = np.random.uniform(low=self.rand_low, high=self.rand_high)
            else:
                self._state = np.array(init_state, dtype=np.float32)
            self.env.model_class.vehicle3dof_InstP.x_ini[:] = self._state
            self.env.model_class.vehicle3dof_InstP.ref_V = np.array(self.ref_curve.V)
            self.env.model_class.vehicle3dof_InstP.ref_fai[:] = np.array(
                self.ref_curve.fai
            )
            self.env.model_class.vehicle3dof_InstP.ref_A[:] = np.array(self.ref_curve.A)
            self.env.model_class.vehicle3dof_InstP.ref_T[:] = np.array(self.ref_curve.T)
            self.env.model_class.vehicle3dof_InstP.done_range[:] = np.array(
                self.done_range
            )
            self.env.model_class.vehicle3dof_InstP.punish_Q[:] = self.Q
            self.env.model_class.vehicle3dof_InstP.punish_R[:] = self.R

        # Reset takes an optional callback
        # This callback will be called after model & parameter initialization and before taking first step.
        state = self.env.reset(callback)
        obs = self.postprocess(state)
        return obs

    def _physics_step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        state, reward, done, info = self.env.step(action)
        self._state = state
        return state, reward, done, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        # Preprocess action
        action_real = self.preprocess(action)
        sum_reward = 0
        for idx in range(self.act_repeat):
            state, reward, done, info = self._physics_step(action_real)
            sum_reward += self.reward_shaping(reward)
            if done:
                sum_reward += self.punish_done
                break
        # Postprocess obs
        obs = self.postprocess(state)
        return obs, sum_reward, done, info

    def preprocess(self, action: np.ndarray) -> Tuple[np.ndarray]:
        action_real = action / self.act_scale
        return action_real

    def postprocess(self, state: np.ndarray) -> Tuple[np.ndarray]:
        ref_y, ref_phi, ref_v = self.ref_curve.cal_reference(state[0])
        obs = np.zeros(self.observation_space.shape)
        obs[0] = state[0]
        obs[1] = state[1] - ref_y
        obs[2] = state[2] - ref_v
        obs[3] = state[3]
        obs[4] = state[4] - ref_phi
        obs[5] = state[5]
        obs[0:6] = obs[0:6] * self.obs_scale
        # Reference position
        if self.use_ref == "Pos":
            x_pre = state[0] + ref_v * self.dt * self.act_repeat * np.linspace(
                1, self.ref_horizon, self.ref_horizon
            )
            y_pre, _, _ = self.ref_curve.cal_reference(x_pre)
            obs_y_pre = (state[1] - y_pre) * self.obs_scale[1]
            obs[6:] = obs_y_pre
        # Reference position and heading angle
        elif self.use_ref == "Both":
            x_pre = state[0] + ref_v * self.dt * self.act_repeat * np.linspace(
                1, self.ref_horizon, self.ref_horizon
            )
            y_pre, phi_pre, _ = self.ref_curve.cal_reference(x_pre)
            obs_y_pre = (state[1] - y_pre) * self.obs_scale[1]
            obs_phi_pre = (state[4] - phi_pre) * self.obs_scale[4]
            obs[6:] = np.concatenate([obs_y_pre, obs_phi_pre])
        return obs

    def reward_shaping(self, origin_reward: float) -> Tuple[float]:
        modified_reward = origin_reward
        if modified_reward <= -self.reward_bound:
            modified_reward = -self.reward_bound
        modified_reward = modified_reward + self.reward_bias
        return modified_reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        pass

    def close(self):
        pass

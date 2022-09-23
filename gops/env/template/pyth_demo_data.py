#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab

import gym
from gym import spaces
from gym.utils import seeding
from gym.wrappers.time_limit import TimeLimit
import numpy as np

gym.logger.setLevel(gym.logger.ERROR)


class PythDemo(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"],
    }

    def __init__(self, **kwargs):
        self.is_adversary = kwargs.get("is_adversary", False)
        self.is_constraint = kwargs.get("is_constraint", False)
        self.info_dict = {}

        # define your custom parameters here

        # self.example_variable = 0

        # define observation space here
        lb_observation = [-np.inf, -np.inf, -np.inf]
        hb_observation = [np.inf, np.inf, np.inf]
        self.observation_space = spaces.Box(
            low=np.array(lb_observation, dtype=np.float32), high=np.array(hb_observation, dtype=np.float32)
        )

        # define action space here
        lb_action = [-1.0, -1.0]
        hb_action = [1.0, 1.0]
        self.action_space = spaces.Box(
            low=np.array(lb_action, dtype=np.float32), high=np.array(hb_action, dtype=np.float32)
        )

        if self.is_adversary:
            # define adversial action space here
            lb_adv_action = [-1.0, -1.0]
            hb_adv_action = [1.0, 1.0]
            self.adv_action_space = spaces.Box(
                low=np.array(lb_adv_action, dtype=np.float32), high=np.array(hb_adv_action, dtype=np.float32)
            )
            self.info_dict["advers"]= {"shape":self.adv_action_space.shape, "dtype":np.float32}
        if self.is_constraint:
            # define constraint dimension here
            self.constraint_dim = 1
            self.info_dict["cons"] = {"shape": self.constraint_dim, "dtype": np.float32}

        # define state dimension here
        # NOTE: Observation may be different from state in some environments! In these cases, if you want to
        #       use the corresponding "*_model.py", you need to define self.state_dim and add relevant information
        #       to self.info
        self.state_dim = (1,)
        self.info_dict["state"] = {"shape": self.state_dim, "dtype": np.float32}

        self.seed()
        self.obs = None

    @property
    def additional_info(self):
        return self.info_dict

    def seed(self, seed=None):
        """
        set random seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray, adv_action=None):
        """
        action: datatype:numpy.ndarray, shape:[action_dim,]
        adv_action: datatype:numpy.ndarray, shape:[adv_action_dim,]
        return:
        self.obs: next observation, datatype:numpy.ndarray, shape:[state_dim]
        reward: reward signal
        done: done signal, datatype: bool
        """

        # define environment transition, reward,  done signal  and constraint function here
        self.obs = self.obs + self.np_random.uniform(low=-0.05, high=0.05, size=(self.observation_space.shape[0],))

        reward = 0
        done = False
        info = {"cons": 0.,"state":self.obs,"advers":0*self.obs}
        return self.obs, reward, done, info

    def reset(self, *, init_obs=None):
        """
        self.obs: initial observation, datatype:numpy.ndarray, shape:[state_dim]
        """
        # define initial state distribution here
        if init_obs is None:
            self.obs = self.np_random.uniform(low=-0.05, high=0.05, size=(self.observation_space.shape[0],))
        else:
            assert self.observation_space.contains(init_obs)
            self.obs = init_obs
        return self.obs

    def render(self, mode="human"):
        pass

    def close(self):
        pass


def env_creator(**kwargs):
    return TimeLimit(PythDemo(**kwargs), 200)


if __name__ == "__main__":
    env = env_creator()
    env.reset()
    action = env.action_space.sample()
    s, r, d, _ = env.step(action)

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#  Creator: iDLab
#  Description: wrappers for `env` and `env model`: transform between
#               env with constraints and envs without constraints

import gym
import numpy as np
import torch

from gops.env.env_wrapper.base import ModelWrapper


class EnvC2U(gym.Wrapper):
    """
    transform an env with constraints to env without constraint by punishing
    the constraint function in the reward function
    """
    def __init__(self, env, punish_factor=10):
        super().__init__(env)
        self.punish_factor = punish_factor


    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        punish = np.sum(self.punish_factor * np.clip(info["constraint"], 0, np.inf))
        reward_new = reward - punish
        return observation, reward_new, done, info


class ModelC2U(ModelWrapper):
    """
    transform an env model with constraints to env model without constraint by punishing
    the constraint function in the reward function
    """
    def __init__(self, model, punish_factor=10):
        super(ModelC2U, self).__init__(model)
        self.model = model
        self.punish_factor = punish_factor
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, beyond_done=None):
        state_next, reward, done, info = self.model.forward(state, action, beyond_done)
        const = info["constraint"].reshape(state_next.shape[0], -1)

        punish = torch.clamp(const, min=0) * self.punish_factor

        reward_new = reward - torch.sum(punish, dim=1)

        return state_next, reward_new, done, info

def main():
    from gops.env.env_archive.pyth_carfollowing_data import PythCarfollowingData
    from gops.env.env_archive.pyth_carfollowing_model import PythCarfollowingModel

    env = EnvC2U(PythCarfollowingData(), punish_factor=10)
    dyn = ModelC2U(PythCarfollowingModel(),punish_factor=10)
    env.reset()
    for i in range(10):
        a = env.action_space.sample()
        env.step(a)

    states = [env.reset() for _ in range(2)]
    actions = [env.action_space.sample() for _ in range(2)]
    states = np.stack(states)
    actions = np.stack(actions)
    states = torch.as_tensor(states, dtype=torch.float32)
    actions = torch.as_tensor(actions, dtype=torch.float32)
    print(states.shape, actions.shape)

    dyn.forward(states, actions)


if __name__ == "__main__":
    main()
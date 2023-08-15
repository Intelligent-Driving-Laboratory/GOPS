from dataclasses import dataclass
from typing import Optional, Any, Union
from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel, ContextModel, EnvModel
from gops.env.env_gen_ocp.pyth_idsim import idSimState, idSimContextState, idSimEnv, get_idsimcontext
from gops.env.env_gen_ocp.pyth_base import State

import numpy as np
import torch
import copy

from idsim_model.model_context import State as ModelState
from idsim_model.model import IdSimModel


@dataclass
class FakeModelContext:
    x: Optional[torch.Tensor] = None


class idSimRobotModel(RobotModel):
    def __init__(self,
        idsim_model: IdSimModel,
    ):
        self.robot_state_dim = 6 + 2 * 2
        #TODO: move action bound to here and add it into state bound? 
        self.robot_state_lower_bound = torch.tensor([-np.inf] * self.robot_state_dim, dtype=torch.float32)
        self.robot_state_upper_bound = torch.tensor([np.inf] * self.robot_state_dim, dtype=torch.float32)
        self.idsim_model = idsim_model
        self.Ts = idsim_model.Ts
        self.vehicle_spec = idsim_model.vehicle_spec
        self.fake_model_context = FakeModelContext()

    def get_next_state(self, robot_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        self.fake_model_context.x = ModelState(
            ego_state = robot_state[..., :-4],
            last_last_action = robot_state[..., -4:-2],
            last_action = robot_state[..., -2:]
        )
        model_state = self.idsim_model.dynamics(self.fake_model_context, action)
        robot_state = torch.concat([model_state.ego_state, model_state.last_last_action, model_state.last_action], dim=-1)
        return robot_state


class idSimContextModel(ContextModel):
    def get_next_state(self, context_state: idSimContextState, action: torch.Tensor) -> idSimContextState:
        next_context_state = copy.copy(context_state)
        next_context_state.t = context_state.t + 1
        return next_context_state


class idSimEnvModel(EnvModel):
    def __init__(
            self,
            env: idSimEnv,
            device: Union[torch.device, str, None] = None,
            **kwargs: Any,
    ):
        super().__init__(
            obs_dim = env.observation_space.shape[0],
            action_dim = env.action_space.shape[0],
            action_lower_bound = env.action_space.low,
            action_upper_bound = env.action_space.high,
            device = device,
            dt = env.config.dt,
        )
        model_config = env.model_config
        self.idsim_model = IdSimModel(env, model_config)
        self.robot_model = idSimRobotModel(idsim_model = self.idsim_model)
        self.context_model = idSimContextModel()
        self.StateClass = idSimState

    def get_obs(self, state: idSimState) -> torch.Tensor:
        return self.idsim_model.observe(get_idsimcontext(state, mode = 'batch'))
        
    def get_reward(self, state: idSimState, action: torch.Tensor, mode: str = "full_horizon") -> torch.Tensor:
        next_state = self.get_next_state(state, action)
        if mode == "full_horizon":
            rewards = self.idsim_model.reward_full_horizon(
                context_full = get_idsimcontext(next_state, mode = mode),
                last_last_action_full = state.robot_state[..., -4:-2], # absolute action
                last_action_full = state.robot_state[..., -2:], # absolute action
                action_full = action # incremental action
            )
        elif mode == "batch":
            rewards = self.idsim_model.reward_nn_state(
                context = get_idsimcontext(next_state, mode = mode),
                last_last_action = state.robot_state[..., -4:-2], # absolute action
                last_action = state.robot_state[..., -2:], # absolute action
                action = action # incremental action
            )
        else:
            raise NotImplementedError
        return rewards[0]

    def get_terminated(self, state: idSimState) -> torch.bool:
        # TODO: temporary implementation
        # only support batched state
        return torch.ones(state.robot_state.shape[0], dtype=torch.bool)
    
    def forward(self, obs, action, done, info):
        state = info["state"]
        next_state = self.get_next_state(state, action)
        next_obs = self.get_obs(next_state)
        reward = self.get_reward(state, action, mode = "batch")
        terminated = self.get_terminated(state)
        next_info = {}
        next_info["state"] = next_state
        return next_obs, reward, terminated, info


def env_model_creator(**kwargs):
    """
    make env model `pyth_idsim_model`
    """
    return idSimEnvModel(**kwargs)

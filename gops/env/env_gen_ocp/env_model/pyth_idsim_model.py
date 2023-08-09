from dataclasses import dataclass
from typing import Optional, Dict, Any
from gops.env.env_gen_ocp.env_model.pyth_base_model import RobotModel, ContextModel, EnvModel
from gops.env.env_gen_ocp.pyth_idsim import idSimState, idSimContextState, idSimEnv
from gops.env.env_gen_ocp.pyth_base import State

import numpy as np
import torch
import copy

from idsim_model.model_context import ModelContext, Parameter
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
            **kwargs: Any,
    ):
        env = kwargs["env"]
        model_config = env.model_config
        self.idsim_model = IdSimModel(env, model_config)
        self.robot_model = idSimRobotModel(idsim_model = self.idsim_model)
        self.context_model = idSimContextModel()
        self.StateClass = idSimState

        self.dt = env.config.dt
        self.action_dim = env.action_space.shape[0]
        self.action_lower_bound = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_upper_bound = torch.tensor(env.action_space.high, dtype=torch.float32)
    
    def get_obs(self, state: idSimState) -> torch.Tensor:
        return self.idsim_model.observe(self._get_idsimcontext(state))
        
    def get_reward(self, state: idSimState, action: torch.Tensor) -> torch.Tensor:
        next_state = self.get_next_state(state, action)
        rewards = self.idsim_model.reward_full_horizon(
                    context_full = self._get_idsimcontext(next_state),
                    last_last_action_full = state.robot_state[..., -4:-2], # absolute action
                    last_action_full = state.robot_state[..., -2:], # absolute action
                    action_full = action # incremental action
                  )
        return rewards[0]

    def get_terminated(self, state: idSimState) -> torch.bool:
        raise NotImplementedError
    
    def _get_idsimcontext(self, state: idSimState) -> ModelContext:
        context = ModelContext(
            x = ModelState(
                ego_state = state.robot_state[..., :-4].unsqueeze(0),
                last_last_action = state.robot_state[..., -4:-2].unsqueeze(0),
                last_action = state.robot_state[..., -2:].unsqueeze(0)
            ),
            p = Parameter(
                ref_param = state.context_state.reference.unsqueeze(0),
                sur_param = state.context_state.constraint.unsqueeze(0),
                light_param = state.context_state.light_param.unsqueeze(0),
                ref_index_param = state.context_state.ref_index_param.unsqueeze(0)
            ),
            t = state.context_state.real_t.unsqueeze(0),
            i = state.context_state.t
        )
        return context


def env_model_creator(**kwargs):
    """
    make env model `pyth_idsim_model`
    """
    return idSimEnvModel(**kwargs)

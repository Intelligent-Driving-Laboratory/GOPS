from dataclasses import dataclass
from typing import Optional, TypeVar, Callable, Tuple
from gops.env.env_gen_ocp.env_model.pyth_model_base import RobotModel, ContextModel, EnvModel, S
from gops.env.env_gen_ocp.pyth_idSim import idSimState, idSimContextState, idSimEnv

import numpy as np
import torch
import copy

from idsim_model.predict_model import ego_predict_model
from idsim_model.model import ModelContext, Parameter, IdSimModel
from idsim_model.model import State as ModelState


@dataclass
class FakeModelContext:
    x: Optional[torch.Tensor] = None


class idSimRobotModel(RobotModel):
    def __init__(self,
        idsim_model: IdSimModel,
    ):
        self.robot_state_dim = 6 + 2 * 2
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
            env: idSimEnv
    ):
        self.idsim_model = IdSimModel(env)
        self.robot_model = idSimRobotModel(idsim_model = self.idsim_model)
        self.context_model = idSimContextModel()
        self.StateClass = idSimState

        self.dt = env.config.dt
        self.action_dim = env.action_space.shape[0]
        self.action_lower_bound = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_upper_bound = torch.tensor(env.action_space.high, dtype=torch.float32)
    
    # def get_constraint(state: idSimState) -> torch.Tensor:
    #     ...

    def get_obs(self, state: idSimState) -> torch.Tensor:
        return self.idsim_model.observe(self._get_idsimcontext(state))
        
    # TODO: Distinguish state reward and action reward
    def get_reward(self, state: idSimState, action: torch.Tensor) -> torch.Tensor:
        next_state = self.get_next_state(state, action)
        reward = self.idsim_model.reward(
            self._get_idsimcontext(next_state),
            action
            )
        return reward

    def get_terminated(self, state: idSimState) -> torch.bool:
        raise NotImplementedError
    
    def _get_idsimcontext(self, state: idSimState) -> ModelContext:
        context = ModelContext(
            x = ModelState(
                ego_state = state.robot_state[..., :-4],
                last_last_action = state.robot_state[..., -4:-2],
                last_action = state.robot_state[..., -2:]
            ),
            p = Parameter(
                ref_param = state.context_state.reference,
                sur_param = state.context_state.constraint,
                light_param = state.context_state.light_param,
                ref_index_param = state.context_state.ref_index_param
            ),
            t = state.context_state.real_t, #
            i = state.context_state.t #
        )
        return context

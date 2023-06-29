from typing import Optional, TypeVar, Callable, Tuple
from gops.env.env_gen_ocp.env_model.pyth_model_base import RobotModel, ContextModel, EnvModel, S
from gops.env.env_gen_ocp.pyth_idSim import idSimState, idSimContextState, idSimEnv

import numpy as np
import torch
import copy

from idsim_model.predict_model import ego_predict_model
from idsim_model.model import ModelContext, Parameter, IdSimModel
from idsim_model.model import State as ModelState


class idSimRobotModel(RobotModel):
    def __init__(self,
        Ts: float = 0.1,
        vehicle_spec: Tuple[float, float, float, float, float, float, float, float] = (1880.0, 1536.7, 1.13, 1.52, -128915.5, -85943.6, 20.0, 0.0)
    ):
        self.Ts = Ts
        self.vehicle_spec = vehicle_spec

    def get_next_state(self, robot_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return ego_predict_model(robot_state, action, self.Ts, self.vehicle_spec)


class idSimContextModel(ContextModel):
    def get_next_state(self, context_state: idSimContextState, action: torch.Tensor) -> idSimContextState:
        next_context_state = copy.copy(context_state)
        next_context_state.t = context_state.t + 1
        next_context_state.last_last_action = context_state.last_action
        next_context_state.last_action = action
        return next_context_state


class idSimEnvModel(EnvModel):
    def __init__(
            self,
            env: idSimEnv
    ):
        self.robot_model = idSimRobotModel(Ts = env.config.dt, vehicle_spec = env.config.vehicle_spec)
        self.context_model = idSimContextModel()
        self.idsim_model = IdSimModel(env)
        self.StateClass = idSimState

        self.dt = env.config.dt
        self.action_dim = env.action_space.shape[0]
        self.state_dim = env.observation_space.shape[0]
        self.action_lower_bound = torch.tensor(env.action_space.low, dtype=torch.float32)
        self.action_upper_bound = torch.tensor(env.action_space.high, dtype=torch.float32)
    
    # def get_constraint(state: idSimState) -> torch.Tensor:
    #     ...

    def get_obs(self, state: idSimState) -> torch.Tensor:
        return self.idsim_model.observe(self._get_idsimcontext(state))
        
    # TODO: Distinguish state reward and action reward
    def get_reward(self, state: idSimState, action: torch.Tensor) -> torch.Tensor:
        increment_action = action - state.context_state.last_action # increment action
        increment_action = (increment_action - self.idsim_model.action_center) / self.idsim_model.action_half_range # normalize
        reward = self.idsim_model.reward(
            self._get_idsimcontext(state),
            increment_action
            )
        return reward

    def get_terminated(self, state: idSimState) -> torch.bool:
        raise NotImplementedError
        return self.idsim_model.done(self._get_idsimcontext(state))
    
    def _get_idsimcontext(self, state: idSimState) -> ModelContext:
        context = ModelContext(
            x = ModelState(
                ego_state = state.robot_state,
                last_last_action = state.context_state.last_last_action,
                last_action = state.context_state.last_action
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

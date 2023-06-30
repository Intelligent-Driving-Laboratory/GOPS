import pathlib
from gops.env.env_gen_ocp.pyth_base import ContextState, State, Robot, Context, Env, stateType
from idsim.envs.env import CrossRoad
from idsim.config import Config
from typing import Tuple
from idsim_model.model import ModelContext
from idsim.utils.fs import TEMP_ROOT
from dataclasses import dataclass
import torch

import numpy as np
import gym


@dataclass
class idSimContextState(ContextState):
    light_param: stateType
    ref_index_param: stateType
    real_t: int


@dataclass
class idSimState(State):
    context_state: idSimContextState
    CONTEXT_STATE_TYPE = idSimContextState


class idSimEnv(CrossRoad, Env):
    def __init__(self, config: Config):
        super(idSimEnv, self).__init__(config)

    def reset(self) -> Tuple[np.ndarray, dict]:
        obs, info = super(idSimEnv, self).reset()
        self._get_state_from_idsim()
        return obs, self._get_info()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = super(idSimEnv, self).step(action)
        self._get_state_from_idsim()
        return obs, reward, terminated, self._get_info()

    def _get_obs(self) -> np.ndarray:
        """abandon this function, use obs from idsim instead"""
        ...

    def _get_reward(self, action: np.ndarray) -> float:
        """abandon this function, use reward from idsim instead"""
        ...
    
    def _get_terminated(self) -> bool:
        """abandon this function, use terminated from idsim instead"""
        ...
    
    def _get_state_from_idsim(self) -> State:
        idsim_context = ModelContext.from_env(self)
        self._state = idSimState(
            robot_state=torch.concat([
                idsim_context.x.ego_state, 
                idsim_context.x.last_last_action, 
                idsim_context.x.last_action],
            dim=-1),
            context_state=idSimContextState(
                reference=idsim_context.p.ref_param, 
                constraint=idsim_context.p.sur_param,
                light_param=idsim_context.p.light_param, 
                ref_index_param=idsim_context.p.ref_index_param,
                real_t = idsim_context.t,
                t = idsim_context.i
            )
        )
        self._state = idSimState.tensor2array(self._state)


def env_creator(**kwargs):
    """
    make env `pyth_veh2dofconti`
    """
    return idSimEnv(kwargs["env_config"])
import pathlib
from dataclasses import dataclass
from typing import Tuple, Dict, Any
from copy import deepcopy
from pathlib import Path
import gym

import numpy as np
import torch
from idsim.config import Config
from idsim.envs.env import CrossRoad
from idsim_model.model_context import ModelContext, Parameter
from idsim_model.model_context import State as ModelState
from idsim_model.params import model_config as default_model_config
from idsim_model.params import ModelConfig
from typing_extensions import Self
from idsim_model.model import IdSimModel

from gops.env.env_gen_ocp.pyth_base import (ContextState, Env, State, stateType)


@dataclass
class idSimContextState(ContextState):
    light_param: stateType
    ref_index_param: stateType
    real_t: stateType


@dataclass
class idSimState(State):
    context_state: idSimContextState
    CONTEXT_STATE_TYPE = idSimContextState

    def get_zero_state(self, batch_size = 1) -> State:
        # only used for batched state
        return idSimState(
            robot_state=np.zeros((batch_size, *self.robot_state.shape[1:]), dtype=np.float32),
            context_state=idSimContextState(
                reference=np.zeros((batch_size, *self.context_state.reference.shape[1:]), dtype=np.float32),
                constraint=np.zeros((batch_size, *self.context_state.constraint.shape[1:]), dtype=np.float32),
                t=np.zeros((batch_size, ), dtype=np.int64),
                light_param=np.zeros((batch_size, *self.context_state.light_param.shape[1:]), dtype=np.float32),
                ref_index_param=np.zeros((batch_size, ), dtype=np.int64),
                real_t=np.zeros((batch_size, ), dtype=np.int64)
            )
        )


class idSimEnv(CrossRoad, Env):
    def __new__(cls: type[Self], env_config: Config, model_config: Dict[str, Any]) -> Self:

        return super(idSimEnv, cls).__new__(cls, env_config)
    
    def __init__(self, env_config: Config, model_config: Dict[str, Any]):
        super(idSimEnv, self).__init__(env_config)
        self.model_config = model_config
        self._state = None
        # get observation_space
        self.model = IdSimModel(self, model_config)
        self.info_dict = {
            "state": self.get_zero_state,
        }
        obs_dim = self._get_obs().shape[0]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    
    def reset(self) -> Tuple[np.ndarray, dict]:
        obs, info = super(idSimEnv, self).reset()
        self._get_state_from_idsim()
        return self._get_obs(), self._get_info(info)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, truncated, info = super(idSimEnv, self).step(action)
        self._get_state_from_idsim()
        done = terminated or truncated
        return self._get_obs(), reward, done, self._get_info(info)
    
    def _get_info(self, info) -> dict:
        info.update({'state': self._state})
        try:
            info['cost'] = self._get_constraint()
        except NotImplementedError:
            pass
        return info
    
    def _get_obs(self) -> np.ndarray:
        """abandon this function, use obs from idsim instead"""
        idsim_context = get_idsimcontext(idSimState.stack([idSimState.array2tensor(self._state)]), mode="batch")
        model_obs = self.model.observe(idsim_context)
        return model_obs.numpy().squeeze(0)

    def _get_reward(self, action: np.ndarray) -> float:
        """abandon this function, use reward from idsim instead"""
        ...
    
    def _get_terminated(self) -> bool:
        """abandon this function, use terminated from idsim instead"""
        ...
    
    def _get_state_from_idsim(self, ref_index_param=None) -> State:
        idsim_context = ModelContext.from_env(self, self.model_config, ref_index_param)
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
                real_t = torch.tensor(idsim_context.t).int(),
                t = torch.tensor(idsim_context.i).int()
            )
        )
        self._state = idSimState.tensor2array(self._state)

    def get_state_from_idsim(self, ref_index_param=None) -> State:
        self._get_state_from_idsim(ref_index_param=ref_index_param)
        return self._state
    
    @property
    def get_zero_state(self) -> np.ndarray:
        if self._state is None:
            self.reset()
        _state = idSimState.stack([self._state])
        return _state.get_zero_state()
    
    @property
    def additional_info(self) -> Dict[str, Dict]:
        return self.info_dict
    

def get_idsimcontext(state: idSimState, mode: str) -> ModelContext:
    if mode == "full_horizon":
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
    elif mode == "batch":
        assert state.context_state.t.unique().shape[0] == 1, "batch mode only support same t"
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
            t = state.context_state.real_t,
            i = state.context_state.t[0]
        )
    else:
        raise NotImplementedError
    return context


def env_creator(**kwargs):
    """
    make env `pyth_idsim`
    """
    env_config = deepcopy(kwargs["env_config"])
    if 'scenario_root' in env_config:
        env_config['scenario_root'] = Path(env_config['scenario_root'])
    env_config = Config.from_partial_dict(env_config)
    if "env_model_config" in kwargs.keys():
        model_config = kwargs["env_model_config"]
    else:
        model_config = default_model_config
    model_config = ModelConfig.from_partial_dict(model_config)
    env = idSimEnv(env_config, model_config)
    return env
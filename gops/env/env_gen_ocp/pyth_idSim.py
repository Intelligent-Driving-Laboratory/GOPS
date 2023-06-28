import pathlib
from gops.env.env_gen_ocp.pyth_base import ContextState, State, Robot, Context, Env, stateType
from idsim.envs.env import CrossRoad
from idsim.config import Config
from typing import Tuple
from idsim_model.model import ModelContext
from idsim.utils.fs import TEMP_ROOT
from dataclasses import dataclass

import numpy as np
import gym


@dataclass
class idSimContextState(ContextState):
    last_last_action: stateType
    last_action: stateType
    light_param: stateType
    ref_index_param: stateType
    real_t: int

@dataclass
class idSimState(State):
    context_state: idSimContextState


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
            robot_state=idsim_context.x.ego_state,
            context_state=idSimContextState(
                reference=idsim_context.p.ref_param, 
                constraint=idsim_context.p.sur_param,
                last_last_action=idsim_context.x.last_last_action,
                last_action=idsim_context.x.last_action,
                light_param=idsim_context.p.light_param, 
                ref_index_param=idsim_context.p.ref_index_param,
                real_t = idsim_context.t,
                t = idsim_context.i
            )
        )


def env_creator(**kwargs):
    """
    make env `pyth_veh2dofconti`
    """
    return idSimEnv(kwargs["env_config"])

MAP_ROOT = pathlib.Path('/home/taoletian/set-new/idsim-scenarios-train')

if __name__ == "__main__":
    arg = {}
    arg["env_config"] = Config(
        use_render=False,
        seed=None,
        actuator="ExternalActuator",
        scenario_reuse=10,
        choose_vehicle_retries=10,
        num_scenarios=20,
        scenario_selector=None,
        scenario_root=MAP_ROOT,
        extra_sumo_args=("--start", "--delay", "200"),
        warmup_time=5.0,
        max_steps=500,
        ignore_traffic_lights=False,
        skip_waiting=False,
        detect_range=40,
        no_done_at_collision=False,
        penalize_collision=True,
        ref_v = 8.0,
        singleton_mode="raise",
        direction_selector=None,
        action_lower_bound=(-3.5, -0.4),
        action_upper_bound=(1.5, 0.4),
        obs_num_surrounding_vehicles={
            'passenger': 4,
            'bicycle': 0,
            'pedestrian': 0,
        },
    )
    env = env_creator(**arg)
    env.reset()
    print(env.observation_space)
    print(env.action_space)
    for i in range(5):
        env.step(np.array([0.0, 0.0]))
        print(env._state)
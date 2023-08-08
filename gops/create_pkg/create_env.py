#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create environments
#  Update Date: 2020-11-10, Yuhang Zhang: add create environments code

from typing import Optional

from gops.env.vector.sync_vector_env import SyncVectorEnv
from gops.env.vector.async_vector_env import AsyncVectorEnv
from gops.env.wrapper.wrapping_utils import wrapping_env
from typing import Callable, Dict, Union
from dataclasses import dataclass, field


@dataclass
class Spec:
    env_id: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)


registry: Dict[str, Spec] = {}



def register(
    env_id: str, entry_point: Union[Callable, str], **kwargs,
):
    global registry

    new_spec = Spec(env_id=env_id, entry_point=entry_point, **kwargs,)

    # print(registry.keys())
    # if new_spec.env_id in registry:
    #     print(f"Overriding environment {new_spec.env_id} already in registry.")
    
    registry[new_spec.env_id] = new_spec


def create_env(
    vector_env_num: Optional[int] = None, vector_env_type: Optional[str] = None, **kwargs
) -> object:
    env_name = kwargs["env_id"]
    spec_ = registry.get(env_name)

    if spec_ is None:
        raise KeyError(f"No registered env with id: {env_name}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        env_creator = spec_.entry_point

    else:
        raise RuntimeError(f"{spec_.env_id} registered but entry_point is not specified")

    # Wrapping the env
    max_episode_steps = kwargs.get("max_episode_steps", None)
    reward_scale = kwargs.get("reward_scale", None)
    reward_shift = kwargs.get("reward_shift", None)
    obs_scale = kwargs.get("obs_scale", None)
    obs_shift = kwargs.get("obs_shift", None)
    obs_noise_type = kwargs.get("obs_noise_type", None)
    obs_noise_data = kwargs.get("obs_noise_data", None)
    gym2gymnasium = kwargs.get("gym2gymnasium", False)

    def env_fn():
        env = env_creator(**kwargs)
        env = wrapping_env(
            env=env,
            max_episode_steps=max_episode_steps,
            reward_shift=reward_shift,
            reward_scale=reward_scale,
            obs_shift=obs_shift,
            obs_scale=obs_scale,
            obs_noise_type=obs_noise_type,
            obs_noise_data=obs_noise_data,
            gym2gymnasium=gym2gymnasium,
        )
        return env

    if vector_env_num is None:
        env = env_fn()
    else:
        env_fns = [env_fn] * vector_env_num
        if vector_env_type == "sync":
            env = SyncVectorEnv(env_fns)
        elif vector_env_type == "async":
            env = AsyncVectorEnv(env_fns)
        else:
            raise ValueError(f"Invalid vector_env_type {vector_env_type}!")

    print("Create environment successfully!")
    return env

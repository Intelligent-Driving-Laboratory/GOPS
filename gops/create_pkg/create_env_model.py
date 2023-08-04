#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create environments
#  Update Date: 2020-11-10, Yuhang Zhang: add create environments code

from gops.env.wrapper.wrapping_utils import wrapping_model
from gops.create_pkg.base import Spec
from typing import Callable, Dict, Union

registry: Dict[str, Spec] = {}


def register(
    id: str, entry_point: Union[Callable, str], max_episode_steps: Optional[int] = None, **kwargs,
):
    global registry

    new_spec = Spec(id=id, entry_point=entry_point, max_episode_steps=max_episode_steps, **kwargs,)

    if new_spec.id in registry:
        print(f"Overriding environment {new_spec.id} already in registry.")
    registry[new_spec.id] = new_spec


def create_env_model(id: str, **kwargs,) -> object:
    spec_ = registry.get(id)

    if spec_ is None:
        raise KeyError(f"No registered env with id: {id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        env_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.id} registered but entry_point is not specified")

    env_model = env_creator(**_kwargs)

    reward_scale = kwargs.get("reward_scale", None)
    reward_shift = kwargs.get("reward_shift", None)
    obs_scale = kwargs.get("obs_scale", None)
    obs_shift = kwargs.get("obs_shift", None)
    clip_obs = kwargs.get("clip_obs", True)
    clip_action = kwargs.get("clip_action", True)
    mask_at_done = kwargs.get("mask_at_done", True)
    env_model = wrapping_model(
        model=env_model,
        reward_shift=reward_shift,
        reward_scale=reward_scale,
        obs_shift=obs_shift,
        obs_scale=obs_scale,
        clip_obs=clip_obs,
        clip_action=clip_action,
        mask_at_done=mask_at_done,
    )
    return env_model

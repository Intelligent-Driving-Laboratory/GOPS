#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create environments
#  Update Date: 2020-11-10, Yuhang Zhang: add create environments code

from gops.env.wrapper.wrapping_utils import wrapping_model
from dataclasses import dataclass, field
from typing import Callable, Dict, Union


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

    # if new_spec.env_id in registry:
    #     print(f"Overriding environment {new_spec.env_id} already in registry.")
    registry[new_spec.env_id] = new_spec


def create_env_model(**kwargs,) -> object:
    env_id = kwargs["env_id"] + "_model"
    spec_ = registry.get(env_id)

    if spec_ is None:
        raise KeyError(f"No registered env with id: {env_id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        env_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.env_id} registered but entry_point is not specified")

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

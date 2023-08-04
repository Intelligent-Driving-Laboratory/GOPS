#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create approximate function module
#  Update Date: 2020-12-13, Hao Sun: add create buffer function

from dataclasses import dataclass, field
from typing import Callable, Dict, Union


from gops.create_pkg.base import Spec

registry: Dict[str, Spec] = {}


def register(
    id: str, entry_point: Callable, **kwargs,
):
    global registry

    new_spec = Spec(id=id, entry_point=entry_point, **kwargs,)

    if new_spec.id in registry:
        print(f"Overriding buffer {new_spec.id} already in registry.")
    registry[new_spec.id] = new_spec


def create_buffer(id: str, **kwargs) -> object:
    spec_ = registry.get(id)

    if spec_ is None:
        raise KeyError(f"No registered buffer with id: {id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        buffer_creator = spec_.entry_point

    else:
        raise RuntimeError(f"{spec_.id} registered but entry_point is not specified")

    if "seed" not in _kwargs or _kwargs["seed"] is None:
        _kwargs["seed"] = 0
    trainer_name = _kwargs.get("trainer", None)
    if trainer_name is None or trainer_name.startswith("on"):
        buf = None
    elif trainer_name.startswith("off_serial"):
        buf = buffer_creator(**_kwargs)
    elif trainer_name.startswith("off_async") or trainer_name.startswith("off_sync"):
        import ray

        buf = [
            ray.remote(num_cpus=1)(buffer_creator).remote(index=idx, **_kwargs) for idx in range(_kwargs["num_buffers"])
        ]
    else:
        raise RuntimeError(f"trainer {trainer_name} not recognized")

    return buf

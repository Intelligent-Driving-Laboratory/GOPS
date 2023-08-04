#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create sampler
#  Update: 2021-03-05, Yuheng Lei: create sampler module


import importlib

from typing import Callable, Dict, Union


from gops.create_pkg.base import Spec


registry: Dict[str, Spec] = {}


def register(
    id: str, entry_point: Union[Callable, str], **kwargs,
):
    global registry

    new_spec = Spec(id=id, entry_point=entry_point, **kwargs,)

    if new_spec.id in registry:
        print(f"Overriding sampler {new_spec.id} already in registry.")
    registry[new_spec.id] = new_spec


def create_sampler(id: str, **kwargs,) -> object:
    spec_ = registry.get(id)

    if spec_ is None:
        raise KeyError(f"No registered sampler with id: {id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        sampler_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.id} registered but entry_point is not specified")

    trainer_name = _kwargs.get("trainer", None)
    if trainer_name is None or trainer_name.startswith("off_serial") or trainer_name.startswith("on_serial"):
        sam = sampler_creator(**_kwargs)
    elif (
        trainer_name.startswith("off_async")
        or trainer_name.startswith("off_sync")
        or trainer_name.startswith("on_sync")
    ):
        import ray

        sam = [
            ray.remote(num_cpus=1)(sampler_creator).remote(index=idx, **_kwargs)
            for idx in range(_kwargs["num_samplers"])
        ]
    else:
        raise RuntimeError(f"trainer {trainer_name} not recognized")

    return sam

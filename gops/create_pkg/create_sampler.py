#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create sampler
#  Update: 2021-03-05, Yuheng Lei: create sampler module


import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Union

from gops.utils.gops_path import sampler_path, underline2camel


@dataclass
class Spec:
    sampler_name: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)


registry: Dict[str, Spec] = {}


def register(
    sampler_name: str, entry_point: Union[Callable, str], **kwargs,
):
    global registry

    new_spec = Spec(sampler_name=sampler_name, entry_point=entry_point, kwargs=kwargs)

    registry[new_spec.sampler_name] = new_spec


# register sampler
sampler_file_list = os.listdir(sampler_path)

for sampler_file in sampler_file_list:
    if sampler_file[-3:] == ".py" and sampler_file[0] != "_" and sampler_file != "base.py":
        sampler_name = sampler_file[:-3]
        mdl = importlib.import_module("gops.trainer.sampler." + sampler_name)
        register(sampler_name=sampler_name, entry_point=getattr(mdl, underline2camel(sampler_name)))


def create_sampler(**kwargs,) -> object:
    sampler_name = kwargs["sampler_name"]
    spec_ = registry.get(sampler_name)

    if spec_ is None:
        raise KeyError(f"No registered sampler with id: {sampler_name}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        sampler_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.sampler_name} registered but entry_point is not specified")

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

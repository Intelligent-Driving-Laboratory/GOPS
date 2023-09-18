#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create approximate function module
#  Update Date: 2020-12-13, Hao Sun: add create buffer function

import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict

from gops.utils.gops_path import buffer_path, underline2camel


@dataclass
class Spec:
    buffer_name: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)


registry: Dict[str, Spec] = {}


def register(
    buffer_name: str, entry_point: Callable, **kwargs,
):
    global registry

    new_spec = Spec(buffer_name=buffer_name, entry_point=entry_point, kwargs=kwargs)

    # if new_spec.buffer_name in registry:
    #     print(f"Overriding buffer {new_spec.buffer_name} already in registry.")
    registry[new_spec.buffer_name] = new_spec


# register buffer
buffer_file_list = os.listdir(buffer_path)

for buffer_file in buffer_file_list:
    if buffer_file[-3:] == ".py" and buffer_file[0] != "_" and buffer_file != "base.py":
        buffer_name = buffer_file[:-3]
        mdl = importlib.import_module("gops.trainer.buffer." + buffer_name)
        register(buffer_name=buffer_name, entry_point=getattr(mdl, underline2camel(buffer_name)))


def create_buffer(**kwargs) -> object:
    buffer_name = kwargs.get("buffer_name", None)
    if buffer_name is None:
        return None
    spec_ = registry.get(buffer_name)

    if spec_ is None:
        raise KeyError(f"No registered buffer with id: {buffer_name}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        buffer_creator = spec_.entry_point

    else:
        raise RuntimeError(f"{spec_.buffer_name} registered but entry_point is not specified")

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

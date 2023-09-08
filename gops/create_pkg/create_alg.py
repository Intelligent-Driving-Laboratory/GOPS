#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create algorithm module
#  Update Date: 2020-12-01, Hao Sun: create algorithm package code

import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict

from gops.utils.gops_path import algorithm_path, underline2camel


@dataclass
class Spec:
    algorithm: str
    entry_point: Callable
    approx_container_cls: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)


registry: Dict[str, Spec] = {}


def register(
    algorithm: str, entry_point: Callable, approx_container_cls: Callable, **kwargs,
):
    global registry

    new_spec = Spec(algorithm=algorithm, entry_point=entry_point, 
                    approx_container_cls=approx_container_cls, kwargs=kwargs)

    # if new_spec.algorithm in registry:
    #     print(f"Overriding algorithm {new_spec.algorithm} already in registry.")
    registry[new_spec.algorithm] = new_spec


# register algorithm
alg_file_list = os.listdir(algorithm_path)

for alg_file in alg_file_list:
    if alg_file[-3:] == ".py" and alg_file[0] != "_" and alg_file != "base.py":
        alg_name = alg_file[:-3]
        mdl = importlib.import_module("gops.algorithm." + alg_name)
        register(
            algorithm=underline2camel(alg_name, first_upper=True),
            entry_point=getattr(mdl, underline2camel(alg_name, first_upper=True)),
            approx_container_cls=getattr(mdl, "ApproxContainer"),
        )


def create_alg(**kwargs) -> object:
    algorithm = kwargs["algorithm"]
    spec_ = registry.get(algorithm)

    if spec_ is None:
        raise KeyError(f"No registered algorithm with id: {algorithm}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        algorithm_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.algorithm} registered but entry_point is not specified")

    if "seed" not in _kwargs or _kwargs["seed"] is None:
        _kwargs["seed"] = 0
    if "cnn_shared" not in _kwargs or _kwargs["cnn_shared"] is None:
        _kwargs["cnn_shared"] = False

    trainer_name = _kwargs.get("trainer", None)
    if (
        trainer_name is None
        or trainer_name.startswith("off_serial")
        or trainer_name.startswith("on_serial")
        or trainer_name.startswith("on_sync")
    ):
        algo = algorithm_creator(**_kwargs)
    elif trainer_name.startswith("off_async") or trainer_name.startswith("off_sync"):
        import ray

        algo = [
            ray.remote(num_cpus=1)(algorithm_creator).remote(index=idx, **_kwargs) for idx in range(_kwargs["num_algs"])
        ]
    else:
        raise RuntimeError(f"trainer {trainer_name} not recognized")

    return algo


def create_approx_contrainer(algorithm: str, **kwargs,) -> object:
    spec_ = registry.get(algorithm)

    if spec_ is None:
        raise KeyError(f"No registered algorithm with id: {algorithm}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if "seed" not in _kwargs or _kwargs["seed"] is None:
        _kwargs["seed"] = 0
    if "cnn_shared" not in _kwargs or _kwargs["cnn_shared"] is None:
        _kwargs["cnn_shared"] = False

    if callable(spec_.approx_container_cls):
        approx_contrainer = spec_.approx_container_cls(**_kwargs)
    else:
        raise RuntimeError(f"{spec_.algorithm} registered but approx_container_cls is not specified")

    return approx_contrainer

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create approximate function module
#  Update Date: 2020-12-26, Hao Sun: add create approximate function

from dataclasses import dataclass, field
from typing import Callable


from gops.create_pkg.base import Spec


def register(
    id: str, entry_point: Callable, **kwargs,
):
    global registry

    new_spec = Spec(id=id, entry_point=entry_point, **kwargs,)

    if new_spec.id in registry:
        print(f"Overriding apprfunc {new_spec.id} already in registry.")
    registry[new_spec.id] = new_spec


def create_alg(id: str, **kwargs,) -> object:
    spec_ = registry.get(id)

    if spec_ is None:
        raise KeyError(f"No registered apprfunc with id: {id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        apprfunc_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.id} registered but entry_point is not specified")

    if "seed" not in _kwargs or _kwargs["seed"] is None:
        _kwargs["seed"] = 0
    if "cnn_shared" not in _kwargs or _kwargs["cnn_shared"] is None:
        _kwargs["cnn_shared"] = False

    apprfunc = apprfunc_creator(**_kwargs)

    return apprfunc

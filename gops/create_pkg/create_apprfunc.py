#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create approximate function module
#  Update Date: 2020-12-26, Hao Sun: add create approximate function

from dataclasses import dataclass, field
from typing import Callable, Dict


@dataclass
class Spec:
    apprfunc: str
    name: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)

registry: Dict[str, Spec] = {}

def register(
    apprfunc: str, name: str, entry_point: Callable, **kwargs,
):
    global registry

    new_spec = Spec(apprfunc=apprfunc, entry_point=entry_point, name=name, **kwargs,)

    # if new_spec.apprfunc in registry:
    #     print(f"Overriding apprfunc {new_spec.apprfunc} - {new_spec.name} already in registry.")
    registry[new_spec.apprfunc + "_" + new_spec.name] = new_spec


def create_apprfunc(**kwargs) -> object:
    apprfunc = kwargs["apprfunc"]
    name = kwargs["name"]
    spec_ = registry.get(apprfunc + "_" + name)

    if spec_ is None:
        raise KeyError(f"No registered apprfunc with id: {apprfunc}_{name}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        apprfunc_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.apprfunc}-{spec_.name} registered but entry_point is not specified")

    apprfunc = apprfunc_creator(**_kwargs)

    return apprfunc

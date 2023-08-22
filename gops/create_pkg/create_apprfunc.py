#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create approximate function module
#  Update Date: 2020-12-26, Hao Sun: add create approximate function

import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict

from gops.utils.gops_path import apprfunc_path


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

    new_spec = Spec(apprfunc=apprfunc, entry_point=entry_point, name=name, kwargs=kwargs)

    # if new_spec.apprfunc in registry:
    #     print(f"Overriding apprfunc {new_spec.apprfunc} - {new_spec.name} already in registry.")
    registry[new_spec.apprfunc + "_" + new_spec.name] = new_spec


# register apprfunc
apprfunc_file_list = os.listdir(apprfunc_path)

for apprfunc_file in apprfunc_file_list:
    if apprfunc_file[-3:] == ".py" and apprfunc_file[0] != "_" and apprfunc_file != "base.py":
        apprfunc_name = apprfunc_file[:-3]
        mdl = importlib.import_module("gops.apprfunc." + apprfunc_name)
        for name in mdl.__all__:
            register(apprfunc=apprfunc_name, name=name, entry_point=getattr(mdl, name))


def create_apprfunc(**kwargs) -> object:
    apprfunc = kwargs["apprfunc"].lower()
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

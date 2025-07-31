#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create optimizer
#  Update: 2024-01-04, Ziang Zheng: create optimizer module


import importlib
import os
import torch
from torch.optim import Adam, SGD
from dataclasses import dataclass, field
from typing import Callable, Dict, Union
from functools import partial

from gops.utils.gops_path import optim_path, underline2camel


@dataclass
class Spec:
    optim_name: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)


registry: Dict[str, Spec] = {}


def register(
        optim_name: str, entry_point: Union[Callable, str], **kwargs,
):
    global registry
    new_spec = Spec(optim_name=optim_name, entry_point=entry_point, kwargs=kwargs)
    registry[new_spec.optim_name] = new_spec


# register optims
optim_file_list = os.listdir(optim_path)

for optim_file in optim_file_list:
    if optim_file[-3:] == ".py" and optim_file[0] != "_" and optim_file != "base.py":
        optim_name = optim_file[:-3]
        mdl = importlib.import_module("gops.optim." + optim_name)
        register(optim_name=optim_name, entry_point=getattr(mdl, underline2camel(optim_name)))

# register torch optims
for t_optim_name in dir(torch.optim):
    if "_" not in t_optim_name:
        register(optim_name=t_optim_name, entry_point=getattr(torch.optim, t_optim_name))


def create_optim(**kwargs, ) -> object:
    optim_name = kwargs["optim_name"] if "optim_name" in kwargs.keys() else "Adam"
    optim_param = kwargs["optim_param"] if "optim_param" in kwargs.keys() else {}
    spec_ = registry.get(optim_name)

    if spec_ is None:
        raise KeyError(f"No registered optim with id: {optim_name}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        optim_creator: torch.optim.Optimizer = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.optim_name} registered but entry_point is not specified or not optimizer")

    # Filter out the keys that are parameters of the class constructor
    valid_args = {k: v for k, v in optim_param.items() if k in optim_creator.__init__.__code__.co_varnames}
    assert "learning_rate" not in valid_args, "Learning rate should not be passed in the optim_param, please use kwargs['learning_rate'] instead"

    # Standard optimizer handling
    def optim_helper(params, **kwargs):
        return optim_creator(params, **kwargs)

    optim_creator_wrap = partial(optim_helper, **valid_args)
    print(f"Create optimizer: {optim_name} with parameters: {valid_args}")
    return optim_creator_wrap

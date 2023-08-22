#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create trainers
#  Update: 2021-03-05, Jiaxin Gao: create trainer module

import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, Union

from gops.utils.gops_path import trainer_path, underline2camel


@dataclass
class Spec:
    trainer: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)


registry: Dict[str, Spec] = {}


def register(
    trainer: str, entry_point: Union[Callable, str], **kwargs,
):
    global registry

    new_spec = Spec(trainer=trainer, entry_point=entry_point, kwargs=kwargs)

    # if new_spec.trainer in registry:
    #     print(f"Overriding trainer {new_spec.trainer} already in registry.")
    registry[new_spec.trainer] = new_spec


# register trainer
trainer_file_list = os.listdir(trainer_path)

for trainer_file in trainer_file_list:
    if trainer_file.endswith("trainer.py"):
        trainer_name = trainer_file[:-3]
        mdl = importlib.import_module("gops.trainer." + trainer_name)
        register(trainer=trainer_name, entry_point=getattr(mdl, underline2camel(trainer_name)))


def create_trainer(alg, sampler, buffer, evaluator, **kwargs,) -> object:
    trainer_name = kwargs["trainer"]
    spec_ = registry.get(trainer_name)

    if spec_ is None:
        raise KeyError(f"No registered trainer with id: {trainer_name}")

    if callable(spec_.entry_point):
        trainer_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.trainer} registered but entry_point is not specified")

    if spec_.trainer.startswith("off"):
        trainer = trainer_creator(alg, sampler, buffer, evaluator, **kwargs)
    elif spec_.trainer.startswith("on"):
        trainer = trainer_creator(alg, sampler, evaluator, **kwargs)
    else:
        raise RuntimeError(f"trainer {spec_.trainer} not recognized")

    return trainer

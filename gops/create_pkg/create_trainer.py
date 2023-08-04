#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create trainers
#  Update: 2021-03-05, Jiaxin Gao: create trainer module

from typing import Callable, Dict, Union


from gops.create_pkg.base import Spec


registry: Dict[str, Spec] = {}


def register(
    id: str, entry_point: Union[Callable, str], **kwargs,
):
    global registry

    new_spec = Spec(id=id, entry_point=entry_point, **kwargs,)

    if new_spec.id in registry:
        print(f"Overriding trainer {new_spec.id} already in registry.")
    registry[new_spec.id] = new_spec


def create_trainer(id: str, algorithm_obj, sampler_obj, buffer_obj, evaluator_obj, **kwargs,) -> object:
    spec_ = registry.get(id)

    if spec_ is None:
        raise KeyError(f"No registered trainer with id: {id}")

    if callable(spec_.entry_point):
        trainer_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.id} registered but entry_point is not specified")

    if spec_.id.startswith("off"):
        trainer = trainer_creator(algorithm_obj, sampler_obj, buffer_obj, evaluator_obj, **kwargs)
    elif spec_.id.startswith("on"):
        trainer = trainer_creator(algorithm_obj, sampler_obj, evaluator_obj, **kwargs)
    else:
        raise RuntimeError(f"trainer {spec_.id} not recognized")

    return trainer

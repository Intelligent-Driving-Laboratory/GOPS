#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create evaluator
#  Update Date: 2020-11-10, Yang Guan: create evaluator module


from ..trainer.evaluator import Evaluator

from typing import Callable, Dict, Union


from gops.create_pkg.base import Spec


registry: Dict[str, Spec] = {}


def register(
    id: str, entry_point: Union[Callable, str], **kwargs,
):
    global registry

    new_spec = Spec(id=id, entry_point=entry_point, **kwargs,)

    if new_spec.id in registry:
        print(f"Overriding evaluator {new_spec.id} already in registry.")
    registry[new_spec.id] = new_spec


def create_evaluator(id: str, **kwargs,) -> object:
    spec_ = registry.get(id)

    if spec_ is None:
        raise KeyError(f"No registered evaluator with id: {id}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        evaluator_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.id} registered but entry_point is not specified")

    trainer_name = _kwargs.get("trainer", None)
    if trainer_name is None or trainer_name.startswith("on_serial") or trainer_name.startswith("off_serial"):
        eva = evaluator_creator(**_kwargs)
    elif (
        trainer_name.startswith("off_async")
        or trainer_name.startswith("off_sync")
        or trainer_name.startswith("on_sync")
    ):
        import ray

        eva = ray.remote(num_cpus=1)(evaluator_creator).remote(**_kwargs)
    else:
        raise RuntimeError(f"trainer {trainer_name} not recognized")

    return eva

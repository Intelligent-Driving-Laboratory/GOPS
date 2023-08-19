#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create evaluator
#  Update Date: 2020-11-10, Yang Guan: create evaluator module


from dataclasses import dataclass, field
from typing import Callable, Dict, Union


@dataclass
class Spec:
    evaluator_name: str
    entry_point: Callable

    # Environment arguments
    kwargs: dict = field(default_factory=dict)

registry: Dict[str, Spec] = {}


def register(
    evaluator_name: str, entry_point: Union[Callable, str], **kwargs,
):
    global registry

    new_spec = Spec(evaluator_name=evaluator_name, entry_point=entry_point, **kwargs,)

    # if new_spec.evaluator_name in registry:
    #     print(f"Overriding evaluator {new_spec.evaluator_name} already in registry.")
    registry[new_spec.evaluator_name] = new_spec


# regist evaluator
from gops.trainer.evaluator import Evaluator
register(evaluator_name="evaluator", entry_point=Evaluator)


def create_evaluator(**kwargs) -> object:
    evaluator_name = kwargs["evaluator_name"]
    spec_ = registry.get(evaluator_name)

    if spec_ is None:
        raise KeyError(f"No registered evaluator with id: {evaluator_name}")

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if callable(spec_.entry_point):
        evaluator_creator = spec_.entry_point
    else:
        raise RuntimeError(f"{spec_.evaluator_name} registered but entry_point is not specified")

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

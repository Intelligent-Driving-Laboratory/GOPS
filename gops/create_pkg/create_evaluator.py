#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create evaluator
#  Update Date: 2020-11-10, Yang Guan: create evaluator module


from ..trainer.evaluator import Evaluator


def create_evaluator(**kwargs):
    evaluator_file_name = kwargs["evaluator_name"].lower()
    trainer = kwargs["trainer"]
    try:
        file = __import__(evaluator_file_name)
    except NotImplementedError:
        raise NotImplementedError("This evaluator does not exist")
    evaluator_name = formatter(evaluator_file_name)

    if hasattr(file, evaluator_name):
        evaluator_cls = getattr(file, evaluator_name)
        if trainer == "off_serial_trainer" or trainer == "on_serial_trainer":
            evaluator = evaluator_cls(**kwargs)
        elif (
            trainer == "off_async_trainer"
            or trainer == "on_sync_trainer"
            or trainer == "off_sync_trainer"
            or trainer == "off_async_trainermix"
        ):
            import ray

            evaluator = ray.remote(num_cpus=1)(Evaluator).remote(**kwargs)
        else:
            raise NotImplementedError("This trainer is not properly defined")
    else:
        raise NotImplementedError("This evaluator is not properly defined")

    print("Create evaluator successfully!")
    return evaluator


def formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res

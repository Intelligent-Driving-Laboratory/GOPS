#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create trainers
#  Update: 2021-03-05, Jiaxin Gao: create trainer module


def create_trainer(alg, sampler, buffer, evaluator, **kwargs):
    trainer_name = kwargs["trainer"]
    try:
        file = __import__(trainer_name)
    except NotImplementedError:
        raise NotImplementedError("This trainer does not exist")

    trainer_name_camel = formatter(trainer_name)

    if hasattr(file, trainer_name_camel):
        trainer_cls = getattr(file, trainer_name_camel)
        if (
            trainer_name == "off_serial_trainer"
            or trainer_name == "off_async_trainer"
            or trainer_name == "off_sync_trainer"
            or trainer_name == "off_async_trainermix"
        ):
            trainer = trainer_cls(alg, sampler, buffer, evaluator, **kwargs)
        elif trainer_name == "on_serial_trainer" or trainer_name == "on_sync_trainer":
            trainer = trainer_cls(alg, sampler, evaluator, **kwargs)
    else:
        raise NotImplementedError("This trainer is not properly defined")
    print("Create trainer successfully!")
    return trainer


def formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res

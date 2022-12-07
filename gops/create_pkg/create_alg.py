#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Create algorithm module
#  Update Date: 2020-12-01, Hao Sun: create algorithm package code

import importlib


def create_alg(**kwargs):
    alg_name = kwargs["algorithm"]
    trainer = kwargs["trainer"]
    alg_file_name = alg_name.lower()
    try:
        module = importlib.import_module("gops.algorithm." + alg_file_name)
    except NotImplementedError:
        raise NotImplementedError("This algorithm does not exist")

    # Serial
    if hasattr(module, alg_name):
        alg_cls = getattr(module, alg_name)
        if (
            trainer == "off_serial_trainer"
            or trainer == "on_serial_trainer"
            or trainer == "on_sync_trainer"
        ):
            alg = alg_cls(**kwargs)
        elif (
            trainer == "off_async_trainer"
            or trainer == "off_async_trainermix"
            or trainer == "off_sync_trainer"
        ):
            import ray

            alg = [
                ray.remote(num_cpus=1)(alg_cls).remote(index=idx, **kwargs)
                for idx in range(kwargs["num_algs"])
            ]
        else:
            raise NotImplementedError("This trainer is not properly defined")
    else:
        raise NotImplementedError("This algorithm is not properly defined")

    print("Create algorithm successfully!")
    return alg

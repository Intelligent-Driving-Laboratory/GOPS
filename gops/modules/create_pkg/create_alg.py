#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Hao SUN
#  Description: Create algorithm
"""

"""

#  Update Date: 2020-12-01, Hao SUN:
import ray
import importlib


def create_alg(**kwargs):
    alg_name = kwargs['algorithm']
    trainer = kwargs['trainer']
    alg_file_name = alg_name.lower()
    try:
        module = importlib.import_module('modules.algorithm.'+alg_file_name)
    except NotImplementedError:
        raise NotImplementedError('This algorithm does not exist')

    # serial
    if hasattr(module, alg_name):
        alg_cls = getattr(module, alg_name)
        if trainer == 'off_serial_trainer' or trainer == 'on_serial_trainer':
            alg = alg_cls(**kwargs)
        elif trainer == 'off_async_trainer':
            alg = [ray.remote(num_cpus=1)(alg_cls).remote(**kwargs)
                   for _ in range(kwargs['num_algs'])]
        else:
            raise NotImplementedError("This trainer is not properly defined")
    else:
        raise NotImplementedError("This algorithm is not properly defined")

    print("Create algorithm successfully!")
    return alg

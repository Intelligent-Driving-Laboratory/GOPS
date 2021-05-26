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
from modules.algorithm.ddpg import DDPG



def create_alg(**kwargs):
    alg_name = kwargs['algorithm']
    trainer = kwargs['trainer']
    alg_file_name = alg_name.lower()
    try:
        file = __import__(alg_file_name)
    except NotImplementedError:
        raise NotImplementedError('This algorithm does not exist')

    # serial
    if hasattr(file, alg_name):
        alg_cls = getattr(file, alg_name)
        if trainer == 'off_serial_trainer':
            alg = alg_cls(**kwargs)
        elif trainer == 'off_async_trainer':
            alg = [ray.remote(num_cpus=1)(DDPG).remote(**kwargs)
                   for _ in range(kwargs['num_algs'])]
        else:
            raise NotImplementedError("This trainer is not properly defined")
    else:
        raise NotImplementedError("This algorithm is not properly defined")

    print("Create algorithm successfully!")
    return alg

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab

from gops.env.resources.intersection.endtoend import CrossroadEnd2end
from gym.wrappers.time_limit import TimeLimit


def env_creator(**kwargs):
    """
    make env `pyth_intersection`
    """
    env = CrossroadEnd2end(training_task="left", num_future_data=0, mode="training")
    return TimeLimit(env, 200)


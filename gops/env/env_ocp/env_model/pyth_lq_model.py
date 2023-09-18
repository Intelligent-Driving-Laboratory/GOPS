#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Linear quadratic control environment model
#  Update Date: 2022-08-12, Yuhang Zhang: create environment model
#  Update Date: 2022-10-24, Yujie Yang: add wrapper


from gops.env.env_ocp.resources import lq_configs
from gops.env.env_ocp.resources.lq_base import LqModel


def env_model_creator(**kwargs):
    """
    make env model `pyth_lq`
    """
    lqc = kwargs.get("lq_config", None)
    if lqc is None:
        config = lq_configs.config_s3a1
    elif isinstance(lqc, str):
        assert hasattr(lq_configs, "config_" + lqc)
        config = getattr(lq_configs, "config_" + lqc)
    elif isinstance(lqc, dict):
        config = lqc

    else:
        raise RuntimeError("lq_config invalid")

    return LqModel(config, kwargs.get("device", None))

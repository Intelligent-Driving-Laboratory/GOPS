from gym.wrappers.time_limit import TimeLimit

from gops.env.resources.linear_quadratic_problem import lq_configs
from gops.env.resources.linear_quadratic_problem.lq_base import LqModel


def env_model_creator(**kwargs):
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

    return LqModel(config)
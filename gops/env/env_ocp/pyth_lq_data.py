from gym.wrappers.time_limit import TimeLimit

from gops.env.env_ocp.resources import lq_configs
from gops.env.env_ocp.resources.lq_base import LqEnv


def env_creator(**kwargs):
    """
    make env `pyth_linearquadratic`
    lq_config should be provided in kwargs
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
    lq_configs.check_lq_config(config)
    return TimeLimit(LqEnv(config, **kwargs), config['max_step'])
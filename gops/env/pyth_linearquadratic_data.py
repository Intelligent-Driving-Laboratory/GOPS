from gym.wrappers.time_limit import TimeLimit

from gops.env.resources.linear_quadratic_problem import lq_configs
from gops.env.resources.linear_quadratic_problem.lq_base import LqEnv, LqModel


def env_creator(**kwargs):
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
    return TimeLimit(LqEnv(config), 200)


def test_lq():
    from gops.create_pkg.create_env import create_env
    from gops.create_pkg.create_env_model import create_env_model
    from gops.env.tools.env_check import check_env0
    from gops.env.tools.model_check import check_model0
    from gops.env.tools.compare_env_model import compare
    config_list = ["s3a1", "s4a1", "s5a1", "s4a2", "s6a3"]
    env_id = "pyth_linearquadratic"
    for config in config_list:
        env = create_env(env_id=env_id, lq_config=config)
        env_model = create_env_model(env_id=env_id, lq_config=config)
        check_env0(env)
        check_model0(env, env_model)
        compare(env, env_model)


if __name__ == "__main__":
    test_lq()
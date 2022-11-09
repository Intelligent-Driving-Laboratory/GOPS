from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic, load_args


env = create_env(**load_args('./results/DDPG/simu_lqs2a1'))

check_dynamic(env)
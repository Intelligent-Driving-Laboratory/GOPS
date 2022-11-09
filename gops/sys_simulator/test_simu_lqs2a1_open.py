from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic, load_args


check_dynamic(env_info=load_args('./results/DDPG/simu_lqs2a1'))
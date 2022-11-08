from gops.create_pkg.create_env import create_env
from gops.env.tools.env_dynamic_checker import check_dynamic


env = create_env(env_id='pyth_idpendulum')

check_dynamic(env)
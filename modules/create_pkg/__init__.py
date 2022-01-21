#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
#  Author: Sun Hao


import os
import sys

py_file_path = os.path.abspath(__file__)
create_pkg_path = os.path.dirname(py_file_path)
modules_path = os.path.dirname(create_pkg_path)

# add apprfunc file to sys path
apprunc_file = 'apprfunc'
apprunc_path = os.path.join(modules_path, apprunc_file)
sys.path.append(apprunc_path)

# add env file to sys path
env_file = 'env'
env_path = os.path.join(modules_path, env_file)
env_archive_path = os.path.join(env_path, 'env_archive')
sys.path.append(env_path)
sys.path.append(env_archive_path)

# add algorithm file to sys path
alg_file = 'algorithm'
alg_path = os.path.join(modules_path, alg_file)
sys.path.append(alg_path)

# add trainer file to sys path
trainer_file = 'trainer'
trainer_path = os.path.join(modules_path, trainer_file)
sys.path.append(trainer_path)

# add buffer file to sys path
buffer_file = 'buffer'
buffer_path = os.path.join(modules_path, trainer_file, buffer_file)
sys.path.append(buffer_path)

# add sampler file to sys path
sampler_file = 'sampler'
sampler_path = os.path.join(modules_path, trainer_file, sampler_file)
sys.path.append(sampler_path)

# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.environ["PYTHONPATH"] = sampler_path + ":" + os.environ.get("PYTHONPATH", "")

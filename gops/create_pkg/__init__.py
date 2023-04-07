#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com


import os
import sys

py_file_path = os.path.abspath(__file__)
create_pkg_path = os.path.dirname(py_file_path)
modules_path = os.path.dirname(create_pkg_path)

# Add apprfunc file to sys path
apprunc_file = "apprfunc"
apprunc_path = os.path.join(modules_path, apprunc_file)
sys.path.append(apprunc_path)

# Add env file to sys path
env_file = "env"
env_path = os.path.join(modules_path, env_file)

env_pyth_path = os.path.join(env_path, "env_pyth")
env_pyth_model_path = os.path.join(env_pyth_path, "env_model")

env_gym_path = os.path.join(env_path, "env_gym")
env_gym_model_path = os.path.join(env_gym_path, "env_model")

env_matlab_path = os.path.join(env_path, "env_matlab")
env_matlab_model_path = os.path.join(env_matlab_path, "env_model")

env_ocp_path = os.path.join(env_path, "env_ocp")
env_ocp_model_path = os.path.join(env_ocp_path, "env_model")

sys.path.append(env_path)

sys.path.append(env_pyth_path)
sys.path.append(env_gym_path)
sys.path.append(env_matlab_path)
sys.path.append(env_ocp_path)

sys.path.append(env_pyth_model_path)
sys.path.append(env_gym_model_path)
sys.path.append(env_matlab_model_path)
sys.path.append(env_ocp_model_path)

# Add algorithm file to sys path
alg_file = "algorithm"
alg_path = os.path.join(modules_path, alg_file)
sys.path.append(alg_path)

# Add trainer file to sys path
trainer_file = "trainer"
trainer_path = os.path.join(modules_path, trainer_file)
sys.path.append(trainer_path)

# Add buffer file to sys path
buffer_file = "buffer"
buffer_path = os.path.join(modules_path, trainer_file, buffer_file)
sys.path.append(buffer_path)

# Add sampler file to sys path
sampler_file = "sampler"
sampler_path = os.path.join(modules_path, trainer_file, sampler_file)
sys.path.append(sampler_path)

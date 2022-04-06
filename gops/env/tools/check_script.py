#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab


import os
import importlib
import gops.create_pkg.create_env as ce
from gops.env.tools.env_check import check_env
from gops.env.tools.model_check import check_model

def get_env_model_files():
    env_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_list = []
    model_list = []
    for f in os.listdir(env_folder_path):
        file_path = os.path.join(env_folder_path, f)
        if os.path.isdir(file_path):
            pass
        elif os.path.splitext(file_path)[1] == ".py" and f.startswith(("gym", "pyth", "simu")):
            if f.endswith("data.py"):
                env_list.append(os.path.splitext(f)[0])
            elif f.endswith("model.py"):
                model_list.append(os.path.splitext(f)[0])

    return env_list, model_list

def main():
    env_list, model_list = get_env_model_files()
    for e in env_list:
        check_env(e[:-5])

    for m in model_list:
        check_model(m[:-6])

if __name__ == '__main__':
    main()
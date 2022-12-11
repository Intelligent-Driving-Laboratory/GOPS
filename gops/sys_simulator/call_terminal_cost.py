#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Build terminal cost for MPC controller
#  Update: 2022-12-11, Shengbo Li: create terminal cost


from gops.utils.common_utils import get_args_from_json
import torch
import os
import argparse
import importlib


# Calling value network to serve as terminal cost
def load_args(log_policy_dir):
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args

def load_apprfunc(log_policy_dir, trained_policy_iteration):
    # Create apprfunc
    args = load_args(log_policy_dir)
    ApproxContainer = getattr(importlib.import_module(f"gops.algorithm.{args['algorithm'].lower()}"), "ApproxContainer")
    networks = ApproxContainer(**args)
    # Load trained apprfunc
    log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(trained_policy_iteration)
    networks.load_state_dict(torch.load(log_path))
    return networks
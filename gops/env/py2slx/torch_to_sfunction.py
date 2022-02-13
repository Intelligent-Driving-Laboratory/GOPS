#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Jie Li
#  Description: transform the policy network saved by PyTorch to S-function
#  Create Date: 2021-05-26
#  Update Date: 2021-09-14

#  General Optimal control Problem Solver (GOPS)


import argparse
import json
import numpy as np
import torch
import sys
import os
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(base_path)

from gops.apprfunc.mlp import DetermPolicy, StochaPolicy
from gops.utils.action_distributions import GaussDistribution, DiracDistribution, ValueDiracDistribution
from gops.utils.utils import get_apprfunc_dict


def network_output(obs):
    ################################################
    # Load parameter dictionary and structure of approximate function
    parser = argparse.ArgumentParser()
    args = vars(parser.parse_args())
    with open('config.json') as f:
        summary_dict = json.load(fp=f)
    for key in summary_dict.keys():
        args[key] = summary_dict[key]
    args.update(dict(action_high_limit=np.array([30.], dtype=np.float32),
                     action_low_limit=np.array([-30.], dtype=np.float32)))

    ################################################
    # Create approximate function
    policy_args = get_apprfunc_dict('policy', args['policy_func_type'], **args)
    if args['policy_func_name'] == 'DetermPolicy':
        policy = DetermPolicy(**policy_args)
    else:
        assert args['policy_func_name'] == 'StochaPolicy', 'policy_func_name ERROR!'
        policy = StochaPolicy(**policy_args)

    ################################################
    # Load the parameter of approximate function
    state_dict = torch.load('policy_285000.pkl')
    state_dict.update(dict(act_high_lim=policy.act_high_lim,
                           act_low_lim=policy.act_low_lim))  # if some keys in state_dict are missing, add it!
    policy.load_state_dict(state_dict)

    ################################################
    # Calculate
    batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype('float32'))
    logits = policy(batch_obs)
    if args['action_type'] == 'continu':
        if args['policy_func_name'] == 'DetermPolicy':
            action_distirbution_cls = DiracDistribution
        else:
            assert args['policy_func_name'] == 'StochaPolicy', 'policy_func_name ERROR!'
            action_distirbution_cls = GaussDistribution
    else:
        assert args['action_type'] == 'discret', 'action_type ERROR!'
        action_distirbution_cls = ValueDiracDistribution
    action_distribution = action_distirbution_cls(logits)
    action = action_distribution.mode()
    action = action.detach().numpy()[0].tolist()
    print('action calculated by python: ', action)

    return action


if __name__ == "__main__":
    obs = [0.00162085, 0.03377522, -0.03513411, 0.0230971]
    network_output(obs)

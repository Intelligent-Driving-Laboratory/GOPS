#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Transform pkl network to onnx version
#  Update: 2023-01-05, Jiaxin Gao: Create codes

import contextlib
import torch, torch.nn as nn
import onnxruntime as ort
import argparse
import os
import sys
from gops.utils.common_utils import get_args_from_json
import numpy as np

py_file_path = os.path.abspath(__file__)
utils_path = os.path.dirname(py_file_path)
gops_path = os.path.dirname(utils_path)
# Add algorithm file to sys path
alg_file = "algorithm"
alg_path = os.path.join(gops_path, alg_file)
sys.path.append(alg_path)


def __load_args(log_policy_dir):
    log_policy_dir = log_policy_dir
    json_path = os.path.join(log_policy_dir, "config.json")
    parser = argparse.ArgumentParser()
    args_dict = vars(parser.parse_args())
    args = get_args_from_json(json_path, args_dict)
    return args

def export_model(model: nn.Module, example_obs: torch.Tensor, path: str):
    with _module_inference(model):
        inference_helper = _InferenceHelper(model)
        torch.onnx.export(inference_helper, example_obs, path, input_names=['input'], output_names=['output'],
                          opset_version=11)

        # inference_helper = torch.jit.trace(inference_helper, example_obs)
        # torch.jit.save(inference_helper, path)


@contextlib.contextmanager
def _module_inference(module: nn.Module):
    training = module.training
    module.train(False)
    yield
    module.train(training)


class _InferenceHelper(nn.Module):
    def __init__(self, model):
        super().__init__()

        from gops.apprfunc.mlp import Action_Distribution

        assert isinstance(model, nn.Module) and isinstance(
            model, Action_Distribution
        ), (
            "The model must inherit from nn.Module and Action_Distribution. "
            f"Got {model.__class__.__mro__}"
        )
        self.model = model

    def forward(self, obs: torch.Tensor):
        obs = obs.unsqueeze(0)
        logits = self.model(obs)
        act_dist = self.model.get_act_dist(logits)
        mode = act_dist.mode()
        return mode.squeeze(0)

def deterministic_policy_export_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    torch.onnx.export(networks.policy, example, output_onnx_model, input_names=['input', "input1"],
                      output_names=['output'], opset_version=11)

def deterministic_stochastic_export_onnx_model(networks, input_dim, policy_dir):

    example = torch.rand(1, input_dim)  # network input dim
    output_onnx_model = policy_dir
    model = networks.policy
    export_model(model, example, output_onnx_model)



if __name__=='__main__':

    # Load trained policy
    log_policy_dir = "../../results/FHADP/230223-220506"
    args = __load_args(log_policy_dir)
    alg_name = args["algorithm"]
    alg_file_name = alg_name.lower()
    file = __import__(alg_file_name)
    ApproxContainer = getattr(file, "ApproxContainer")
    networks = ApproxContainer(**args)

    # Load trained policy
    log_path = log_policy_dir + "/apprfunc/apprfunc_{}.pkl".format(44000)  # network position
    networks.load_state_dict(torch.load(log_path))
    networks.eval()

    # create onnx model
    ### example of deterministic policy FHADP algorithm
    input_dim = 90
    policy_dir = '../../transform_onnx_network/network_fhadp_0224_avoid_left.onnx'
    deterministic_policy_export_onnx_model(networks, input_dim, policy_dir)

    # ### example of stochastic policy sac algorithm
    # input_dim = 50
    # policy_dir = '../../transform_onnx_network/network_sac_ziqing.onnx'
    # deterministic_stochastic_export_onnx_model(networks, input_dim, policy_dir)

    # load onnx model for test
    ### example of deterministic policy FHADP algorithm
    ort_session = ort.InferenceSession("../../transform_onnx_network/network_fhadp_0224_avoid_left.onnx")
    example1 = np.random.randn(1, 90).astype(np.float32)
    inputs = {ort_session.get_inputs()[0].name: example1, ort_session.get_inputs()[1].name: np.ones(1).astype(np.int64)}
    outputs = ort_session.run(None, inputs)
    print(outputs[0])
    action = networks.policy(torch.tensor(example1))
    print(action)

    # ### example of stochastic policy sac algorithm
    # ort_session = ort.InferenceSession("../../transform_onnx_network/network_sac_ziqing.onnx")
    # example1 = np.random.randn(1, 50).astype(np.float32)
    # inputs = {ort_session.get_inputs()[0].name: example1}
    # outputs = ort_session.run(None, inputs)
    # print(outputs)
    # action = networks.policy(torch.tensor(example1))
    # act_dist = model.get_act_dist(action).mode()
    # print(act_dist)
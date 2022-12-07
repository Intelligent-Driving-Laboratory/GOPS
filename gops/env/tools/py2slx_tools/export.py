#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com

#  Description: Put GOPS policy back into Simulink for closed-loop validation tool
#  Update Date: 2022-07-011, Yuxuan Jiang: Create policy check and export modular

import contextlib
import torch, torch.nn as nn
import torch.jit


def check_jit_compatibility(model: nn.Module, example_obs: torch.Tensor):
    with _module_inference(model):
        inference_helper = _InferenceHelper(model)
        try:
            torch.jit.trace(inference_helper, example_obs)
        except Exception as e:
            raise RuntimeError(
                "The model cannot be compiled into a trace module."
            ) from e


def export_model(model: nn.Module, example_obs: torch.Tensor, path: str):
    with _module_inference(model):
        inference_helper = _InferenceHelper(model)
        inference_helper = torch.jit.trace(inference_helper, example_obs)
        torch.jit.save(inference_helper, path)


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

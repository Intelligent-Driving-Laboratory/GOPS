import contextlib

import numpy as np
import torch, torch.nn as nn
import torch.jit


def check_jit_compatibility(model, example_obs):
    with _module_inference(model):
        inference_helper = _InferenceHelper(model)
        try:
            torch.jit.trace(inference_helper, example_obs)
        except Exception as e:
            raise RuntimeError("The model cannot be compiled into a trace module.") from e


def export_model(model, example_obs, path):
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
        assert isinstance(model, nn.Module) and isinstance(model, Action_Distribution), \
            "The model must inherit from nn.Module and Action_Distribution. " \
            f"Got {model.__class__.__mro__}"
        self.model = model

    def forward(self, obs: torch.Tensor):
        obs = obs.unsqueeze(0)
        logits = self.model(obs)
        act_dist = self.model.get_act_dist(logits)
        mode = act_dist.mode()
        return mode.squeeze(0)


if __name__ == "__main__":
    import gops.create_pkg  # For PYTHONPATH
    from gops.apprfunc.mlp import DetermPolicy, StochaPolicy
    from gops.utils.action_distributions import GaussDistribution, DiracDistribution

    model = StochaPolicy(
        obs_dim=20,
        act_dim=5,
        hidden_sizes=(64, 64),
        hidden_activation="tanh",
        output_activation="linear",
        act_high_lim=np.ones(5, dtype=np.float32),
        act_low_lim=np.ones(5, dtype=np.float32) * -1.0,
        # DetermPolicy specific
        # action_distirbution_cls=DiracDistribution,
        # StochaPolicy specific
        action_distirbution_cls=GaussDistribution,
        std_sype="mlp_shared",
        min_log_std=-20.0,
        max_log_std=2.0,
    )
    example_obs = torch.randn(20)
    
    check_jit_compatibility(model, example_obs)
    export_model(model, example_obs, "./model.pt")

    # Test usability
    inference_helper = torch.jit.load("./model.pt")
    with torch.no_grad():
        act = inference_helper(example_obs)
    print("Raw: ", model(example_obs))
    print("JIT: ", act)

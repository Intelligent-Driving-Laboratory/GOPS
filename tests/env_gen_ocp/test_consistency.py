import numpy as np
import torch
import pytest

from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_env_model import create_env_model

"""
    Add new test cases in the following two lists, each test case is a dict with keys:
    For old and new env consistency tests:
        "env_old_id": old env id
        "env_new_id": new env id
    For env and model consistency tests:
        "env_id": env id
    For all tests:
        "rtol": relative tolerance for np.isclose
        "atol": absolute tolerance for np.isclose
        "step": number of steps to test
        "seed": seed for env,
    last four keys among which are optional
"""
raw_test_cases_env_old_vs_new = [
    {
        "env_old_id": "pyth_veh2dofconti",
        "env_new_id": "veh2dof_tracking",
    },
    {
        "env_old_id": "pyth_veh3dofconti",
        "env_new_id": "veh3dof_tracking",
    },
    {
        "env_old_id": "pyth_veh2dofconti_errcstr",
        "env_new_id": "veh2dof_tracking_error",
    },
    {
        "env_old_id": "pyth_veh3dofconti_errcstr",
        "env_new_id": "veh3dof_tracking_error",
    },
    {
        "env_old_id": "pyth_veh3dofconti_detour",
        "env_new_id": "veh3dof_tracking_detour",
    },
    {
        "env_old_id": "pyth_veh3dofconti_surrcstr",
        "env_new_id": "veh3dof_tracking_surrcstr",
    },
    {
        "env_old_id": "pyth_lq",
        "env_new_id": "lq_control",
    },
    {
        "env_old_id": "gym_cartpoleconti",
        "env_new_id": "cartpoleconti",
    },
]

raw_test_cases_env_vs_model = [
    {
        "env_id": "veh2dof_tracking",
    },
    {
        "env_id": "veh3dof_tracking",
    },
    {
        "env_id": "veh2dof_tracking_error",
    },
    {
        "env_id": "veh3dof_tracking_error",
    },
    {
        "env_id": "veh3dof_tracking_detour",
    },
    {
        "env_id": "veh3dof_tracking_surrcstr",
    },
    {
        "env_id": "idpendulum",
    },
    {
        "env_id": "lq_control",
    },
    {
        "env_id": "cartpoleconti",
    },
    {
        "env_id": "pendulum",
    },
    {
        "env_id": "quadrotor_1dof_tracking_stablization"
    }
]

DEFAULT_PARAMS = {
    "rtol": 1e-5,
    "atol": 1e-6,
    "step": 10,
    "seed": 0,
}

@pytest.fixture
def test_cases_env_old_vs_new(request):
    return {**DEFAULT_PARAMS, **request.param}

@pytest.fixture
def test_cases_env_vs_model(request):
    return {**DEFAULT_PARAMS, **request.param}

@pytest.mark.parametrize("test_cases_env_old_vs_new", raw_test_cases_env_old_vs_new, indirect=True)
def test_env_old_vs_new_consistency(test_cases_env_old_vs_new):
    env_old_id = test_cases_env_old_vs_new["env_old_id"]
    env_new_id = test_cases_env_old_vs_new["env_new_id"]
    rtol = test_cases_env_old_vs_new["rtol"]
    atol = test_cases_env_old_vs_new["atol"]
    step = test_cases_env_old_vs_new["step"]
    seed = test_cases_env_old_vs_new["seed"]
    env_old = create_env(env_old_id)
    env_new = create_env(env_new_id)
    env_old.seed(seed)
    env_new.seed(seed)
    env_old.action_space.seed(seed)
    env_new.action_space.seed(seed)

    obs_old, _ = env_old.reset()
    obs_new, _ = env_new.reset()
    assert np.isclose(obs_old, obs_new, rtol=rtol, atol=atol).all(), "obs not close on reset!"

    for i in range(step):
        action = env_old.action_space.sample()
        next_obs_old, reward_old, done_old, info_old = env_old.step(action)
        next_obs_new, reward_new, done_new, info_new = env_new.step(action)
        assert np.isclose(next_obs_old, next_obs_new, rtol=rtol, atol=atol).all(), \
            f"obs not close on step {i}!"
        assert np.isclose(reward_old, reward_new, rtol=rtol, atol=atol), \
            f"reward not close on step {i}!"
        assert done_old == done_new, f"done not equal on step {i}!"
        if "constraint" in info_old:
            constraint_old = info_old["constraint"]
            constraint_new = info_new["constraint"]
            assert np.isclose(constraint_old, constraint_new, rtol=rtol, atol=atol).all(), \
                f"constraint not close on step {i}!"
        if done_old:
            break

    print(f'Tested {i + 1} steps! New env is consistent with old env!')

@pytest.mark.parametrize("test_cases_env_vs_model", raw_test_cases_env_vs_model, indirect=True)
def test_env_vs_model_consistency(test_cases_env_vs_model):
    env_id = test_cases_env_vs_model["env_id"]
    rtol = test_cases_env_vs_model["rtol"]
    atol = test_cases_env_vs_model["atol"]
    step = test_cases_env_vs_model["step"]
    seed = test_cases_env_vs_model["seed"]
    env = create_env(env_id)
    model = create_env_model(env_id)
    env.seed(seed)
    env.action_space.seed(seed)

    obs_env, _ = env.reset()
    state = env.state.batch(1).array2tensor()
    obs_model = model.get_obs(state)
    assert np.isclose(obs_env, obs_model.numpy(), rtol=rtol, atol=atol).all(), "obs not close on reset!"

    for i in range(step):
        action = env.action_space.sample()
        next_state = model.get_next_state(state, torch.from_numpy(action).unsqueeze(0))
        next_obs_model = model.get_obs(next_state)
        reward_model = model.get_reward(state, torch.from_numpy(action).unsqueeze(0))
        done_model = model.get_terminated(next_state)
        next_obs_env, reward_env, done_env, info_env = env.step(action)
        assert np.isclose(next_obs_env, next_obs_model.numpy(), rtol=rtol, atol=atol).all(), \
            f"obs not close on step {i}!"
        if not done_env:
            # skip reward check at done because reward_env may include penalty while reward_model does not
            assert np.isclose(reward_env, reward_model.item(), rtol=rtol, atol=atol), \
                f"reward not close on step {i}!"
        assert done_env == done_model.item(), f"done not equal on step {i}!"
        if "constraint" in info_env:
            constraint_env = info_env["constraint"]
            constraint_model = model.get_constraint(next_state)
            assert np.isclose(constraint_env, constraint_model.numpy(), rtol=rtol, atol=atol).all(), \
                f"constraint not close on step {i}!"
        if done_env:
            break
        state = next_state

    print(f'Tested {i + 1} steps! Env model is consistent with env!')

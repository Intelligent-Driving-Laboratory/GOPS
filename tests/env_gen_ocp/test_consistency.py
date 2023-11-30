import numpy as np
import torch
import pytest

from gops.env.env_ocp.resources.lq_base import LqEnv
from gops.env.env_gen_ocp.lq_control import LqControl
from gops.env.env_gen_ocp.env_model.lq_control_model import LqControlModel

from gops.env.env_ocp.pyth_veh2dofconti import SimuVeh2dofconti
from gops.env.env_gen_ocp.veh2dof_tracking import Veh2DoFTracking
from gops.env.env_gen_ocp.env_model.veh2dof_tracking_model import Veh2DoFTrackingModel

from gops.env.env_ocp.pyth_veh3dofconti import SimuVeh3dofconti
from gops.env.env_gen_ocp.veh3dof_tracking import Veh3DoFTracking
from gops.env.env_gen_ocp.env_model.veh3dof_tracking_model import Veh3DoFTrackingModel

from gops.env.env_ocp.pyth_veh2dofconti_errcstr import SimuVeh2dofcontiErrCstr
from gops.env.env_gen_ocp.veh2dof_tracking_error import Veh2DoFTrackingError
from gops.env.env_gen_ocp.env_model.veh2dof_tracking_error_model import Veh2DoFTrackingErrorModel

from gops.env.env_ocp.pyth_veh3dofconti_errcstr import SimuVeh3dofcontiErrCstr
from gops.env.env_gen_ocp.veh3dof_tracking_error import Veh3DoFTrackingError
from gops.env.env_gen_ocp.env_model.veh3dof_tracking_error_model import Veh3DoFTrackingErrorModel

from gops.env.env_ocp.pyth_veh3dofconti_detour import SimuVeh3dofcontiDetour
from gops.env.env_gen_ocp.veh3dof_tracking_detour import Veh3DoFTrackingDetour
from gops.env.env_gen_ocp.env_model.veh3dof_tracking_detour_model import Veh3DoFTrackingDetourModel

from gops.env.env_ocp.pyth_veh3dofconti_surrcstr import SimuVeh3dofcontiSurrCstr
from gops.env.env_gen_ocp.veh3dof_tracking_surrcstr import Veh3DoFTrackingSurrCstr
from gops.env.env_gen_ocp.env_model.veh3dof_tracking_surrcstr_model import Veh3DoFTrackingSurrCstrModel

from gops.env.env_ocp.pyth_idpendulum import PythInverteddoublependulum
from gops.env.env_gen_ocp.idpendulum import Inverteddoublependulum
from gops.env.env_gen_ocp.env_model.idpendulum_model import IdpendulumMdl

from gops.env.env_gen_ocp.env_model.cartpoleconti_model import CartpolecontiMdl
from gops.env.env_gen_ocp.cartpoleconti import Cartpoleconti
from gops.env.env_gym.gym_cartpoleconti import _GymCartpoleconti

"""
    Add new test cases in the following two lists, each test case is a dict with keys:
        "env_old_cls": old env class / "env_cls": env class
        "env_new_cls": new env class / "model_cls": model class
        "rtol": relative tolerance for np.isclose
        "atol": absolute tolerance for np.isclose
        "step": number of steps to test
        "seed": seed for env,
    last four keys among which are optional
"""
raw_test_cases_env_old_vs_new = [
    {
        "env_old_cls": SimuVeh2dofconti,
        "env_new_cls": Veh2DoFTracking,
    },
    {
        "env_old_cls": SimuVeh3dofconti,
        "env_new_cls": Veh3DoFTracking,
    },
    {
        "env_old_cls": SimuVeh2dofcontiErrCstr,
        "env_new_cls": Veh2DoFTrackingError,
    },
    {
        "env_old_cls": SimuVeh3dofcontiErrCstr,
        "env_new_cls": Veh3DoFTrackingError,
    },
    {
        "env_old_cls": SimuVeh3dofcontiDetour,
        "env_new_cls": Veh3DoFTrackingDetour,
    },
    {
        "env_old_cls": SimuVeh3dofcontiSurrCstr,
        "env_new_cls": Veh3DoFTrackingSurrCstr,
    },
    {
        "env_old_cls": LqEnv,
        "env_new_cls": LqControl,
    },
    {
        "env_old_cls": _GymCartpoleconti,
        "env_new_cls": Cartpoleconti,
    },
]

raw_test_cases_env_vs_model = [
    {
        "env_cls": Veh2DoFTracking,
        "model_cls": Veh2DoFTrackingModel,
    },
    {
        "env_cls": Veh3DoFTracking,
        "model_cls": Veh3DoFTrackingModel,
    },
    {
        "env_cls": Veh2DoFTrackingError,
        "model_cls": Veh2DoFTrackingErrorModel,
    },
    {
        "env_cls": Veh3DoFTrackingError,
        "model_cls": Veh3DoFTrackingErrorModel,
    },
    {
        "env_cls": Veh3DoFTrackingDetour,
        "model_cls": Veh3DoFTrackingDetourModel,
    },
    {
        "env_cls": Veh3DoFTrackingSurrCstr,
        "model_cls": Veh3DoFTrackingSurrCstrModel,
    },
    {
        "env_cls": Inverteddoublependulum,
        "model_cls": IdpendulumMdl,
    },
    {
        "env_cls": LqControl,
        "model_cls": LqControlModel,  
    },
    {
        "env_cls": Cartpoleconti,
        "model_cls": CartpolecontiMdl,
    },
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
    env_old_cls = test_cases_env_old_vs_new["env_old_cls"]
    env_new_cls = test_cases_env_old_vs_new["env_new_cls"]
    rtol = test_cases_env_old_vs_new["rtol"]
    atol = test_cases_env_old_vs_new["atol"]
    step = test_cases_env_old_vs_new["step"]
    seed = test_cases_env_old_vs_new["seed"]
    env_old = env_old_cls()
    env_new = env_new_cls()
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
            print(constraint_old-constraint_new)
            assert np.isclose(constraint_old, constraint_new, rtol=rtol, atol=atol).all(), \
                f"constraint not close on step {i}!"
        if done_old:
            break

    print(f'Tested {i + 1} steps! New env is consistent with old env!')

@pytest.mark.parametrize("test_cases_env_vs_model", raw_test_cases_env_vs_model, indirect=True)
def test_env_vs_model_consistency(test_cases_env_vs_model):
    env_cls = test_cases_env_vs_model["env_cls"]
    model_cls = test_cases_env_vs_model["model_cls"]
    rtol = test_cases_env_vs_model["rtol"]
    atol = test_cases_env_vs_model["atol"]
    step = test_cases_env_vs_model["step"]
    seed = test_cases_env_vs_model["seed"]
    env = env_cls()
    model = model_cls()
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

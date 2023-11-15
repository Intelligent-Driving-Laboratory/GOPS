import gym
import numpy as np
import torch
from gops.env.env_gen_ocp.pyth_base import Env
from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel


def check_env_old_new_consistency(
    env_old: gym.Env,
    env_new: Env,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    step: int = 10,
    seed: int = 0,
):
    env_old.seed(seed)
    env_new.seed(seed)
    env_old.action_space.seed(seed)
    env_new.action_space.seed(seed)

    obs_old, _ = env_old.reset()
    obs_new, _ = env_new.reset()
    assert np.isclose(obs_old, obs_new, rtol=rtol, atol=atol).all(), "obs not close on reset!"

    for i in range(step):
        action = env_old.action_space.sample()
        next_obs_old, reward_old, done_old, _ = env_old.step(action)
        next_obs_new, reward_new, done_new, _ = env_new.step(action)
        assert np.isclose(next_obs_old, next_obs_new, rtol=rtol, atol=atol).all(), \
            f"obs not close on step {i}!"
        assert np.isclose(reward_old, reward_new, rtol=rtol, atol=atol), \
            f"reward not close on step {i}!"
        assert done_old == done_new, f"done not equal on step {i}!"
        if done_old:
            break

    print(f'Tested {i + 1} steps! New env is consistent with old env!')


def check_env_model_consistency(
    env: Env,
    model: EnvModel,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    step: int = 10,
    seed: int = 0,
):
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
        state = next_state
        next_obs_env, reward_env, done_env, _ = env.step(action)
        assert np.isclose(next_obs_env, next_obs_model.numpy(), rtol=rtol, atol=atol).all(), \
            f"obs not close on step {i}!"
        if not done_env:
            # skip reward check at done because reward_env may include penalty while reward_model does not
            assert np.isclose(reward_env, reward_model.item(), rtol=rtol, atol=atol), \
                f"reward not close on step {i}!"
        assert done_env == done_model.item(), f"done not equal on step {i}!"
        if done_env:
            break

    print(f'Tested {i + 1} steps! Env model is consistent with env!')

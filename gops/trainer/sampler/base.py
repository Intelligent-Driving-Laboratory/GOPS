#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: base class for samplers
#  Update: 2023-07-22, Zhilong Zheng: create BaseSampler


from abc import ABCMeta, abstractmethod
from typing import List, NamedTuple, Tuple, Union
import time

import numpy as np
import torch

from gops.create_pkg.create_env import create_env
from gops.env.vector.vector_env import VectorEnv
from gops.utils.common_utils import set_seed
from gops.utils.explore_noise import GaussNoise, EpsilonGreedy
from gops.utils.tensorboard_setup import tb_tags


class Experience(NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: float
    done: bool
    info: dict
    next_obs: np.ndarray
    next_info: dict
    logp: float


class BaseSampler(metaclass=ABCMeta):
    def __init__(
        self, 
        sample_batch_size,
        index=0, 
        noise_params=None,
        **kwargs
    ):
        self.env = create_env(**kwargs)
        _, self.env = set_seed(kwargs["trainer"], kwargs["seed"], index + 200, self.env)  #? seed here?
        alg_name = kwargs["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        networks = ApproxContainer(**kwargs)
        self.networks = networks
        self.noise_params = noise_params
        self.sample_batch_size = sample_batch_size
        if isinstance(self.env, VectorEnv):
            self._is_vector = True
            self.num_envs = self.env.num_envs
            self.horizon = self.sample_batch_size // self.num_envs
        else:
            self._is_vector = False
            self.num_envs = 1
            self.horizon = self.sample_batch_size
        self.action_type = kwargs["action_type"]
        self.reward_scale = 1.0  #? why hard-coded?
        if self.noise_params is not None:
            if self.action_type == "continu":
                self.noise_processor = GaussNoise(**self.noise_params)
            elif self.action_type == "discret":
                self.noise_processor = EpsilonGreedy(**self.noise_params)
        
        self.total_sample_number = 0
        self.obs, self.info = self.env.reset()
        if self._is_vector:
            # convert a dict of batched data to a list of dict of unbatched data
            # e.g. next_info = {"a": [1, 2, 3], "b": [4, 5, 6]} ->
            #      unbatched_infos = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
            # ref: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
            self.info = [dict(zip(self.info, t)) for t in zip(*self.info.values())] if self.info else [{}] * self.num_envs

    def load_state_dict(self, state_dict):
        self.networks.load_state_dict(state_dict)

    def sample(self) -> Tuple[Union[List[Experience], dict], dict]:
        self.total_sample_number += self.sample_batch_size
        tb_info = dict()
        start_time = time.perf_counter()

        data = self._sample()

        end_time = time.perf_counter()
        tb_info[tb_tags["sampler_time"]] = (end_time - start_time) * 1000
        return data, tb_info
    
    @abstractmethod
    def _sample(self) -> Union[List[Experience], dict]:
        pass

    def get_total_sample_number(self) -> int:
        return self.total_sample_number
    
    def _step(self) -> List[Experience]:
        # take action using behavior policy
        if not self._is_vector:
            batch_obs = torch.from_numpy(
                np.expand_dims(self.obs, axis=0).astype("float32")
            )
        else:
            batch_obs = torch.from_numpy(self.obs.astype("float32"))
        logits = self.networks.policy(batch_obs)
        action_distribution = self.networks.create_action_distributions(logits)
        action, logp = action_distribution.sample()

        if self._is_vector:
            action = action.detach().numpy()
            logp = logp.detach().numpy()
        else:
            action = action.detach()[0].numpy()
            logp = logp.detach()[0].numpy()

        if self.noise_params is not None:
            action = self.noise_processor.sample(action)
        
        if self.action_type == "continu":
            action_clip = action.clip(
                self.env.action_space.low, self.env.action_space.high
            )
        else:
            action_clip = action
        
        # interact with environment
        if self._is_vector:
            next_obs, reward, terminated, truncated, next_info = self.env.step(action_clip)
            # For vector env, next_obs, reward, terminated, truncated, and next_info are batched data,
            # and vector env will automatically reset the environment when terminated or truncated is True,
            # So we need to get real final observation and info from next_info.

            # Get real final observation
            if "final_observation" in next_info.keys():
                # get the index where next_info["_final_observation"] is True
                index = np.where(next_info["_final_observation"])[0]
                next_obs[index, :] = np.stack(next_info["final_observation"][index])
                
            
            # convert a dict of batched data to a list of dicts of unbatched data
            # e.g. next_info = {"a": [1, 2, 3], "b": [4, 5, 6]} ->
            #      unbatched_infos = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
            # ref: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
            unbatched_infos = [dict(zip(next_info, t)) for t in zip(*next_info.values())]
            
            # Get real final info
            if "final_info" in next_info.keys():
                for i in index:
                    unbatched_infos[i] = next_info["final_info"][i]

            experiences = [Experience(*e) for e in zip(self.obs, action, reward, terminated, self.info, next_obs, unbatched_infos, logp)]

            self.obs = next_obs
            self.info = unbatched_infos

            return experiences
            
        else:
            next_obs, reward, done, next_info = self.env.step(action_clip)

            # TODO: deprecate this after change to gymnasium
            if "TimeLimit.truncated" not in next_info.keys():
                next_info["TimeLimit.truncated"] = False
            if next_info["TimeLimit.truncated"]:
                done = False
        
            experience = Experience(
                obs=self.obs.copy(),
                action=action,
                reward=self.reward_scale * reward,
                done=done,
                info=self.info,
                next_obs=next_obs.copy(),
                next_info=next_info,
                logp=logp,
            )
            
            self.obs = next_obs
            self.info = next_info
            if done or next_info["TimeLimit.truncated"]:
                self.obs, self.info = self.env.reset()

            return [experience]

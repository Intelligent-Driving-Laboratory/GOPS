#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab

from gops.env.resources.intersection.endtoend import CrossroadEnd2end
from gym.wrappers.time_limit import TimeLimit


def env_creator(**kwargs):
    env = CrossroadEnd2end(training_task="left", num_future_data=0, mode="training")
    return TimeLimit(env, 200)


if __name__ == '__main__':
    import numpy as np
    env = env_creator()
    obs = env.reset()
    print(env.observation_space, env.action_space)
    i = 0
    while i < 100000:
        for j in range(200):
            i += 1
            # action=2*np.random.random(2)-1
            if obs[4] < -18:
                action = np.array([0, 1], dtype=np.float32)
            elif obs[3] <= -18:
                action = np.array([0, 0], dtype=np.float32)
            else:
                action = np.array([0.2, 0.33], dtype=np.float32)
            obs, reward, done, info = env.step(action)
            obses, actions = obs[np.newaxis, :], action[np.newaxis, :]
            # extract infos for each kind of participants
            start = 0
            end = start + env.ego_info_dim + env.per_tracking_info_dim * (env.num_future_data + 1)
            obses_ego = obses[:, start:end]

            start = end
            end = start + env.per_veh_info_dim * env.veh_num
            obses_veh = obses[:, start:end]

            obses_veh = np.reshape(obses_veh, [-1, env.per_veh_info_dim])

            env.render()
            if done:
                break
        done = 0
        obs = env.reset()
        env.render()
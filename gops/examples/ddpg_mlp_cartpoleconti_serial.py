#   Copyright (c) Intelligent Driving Lab(iDLab), Tsinghua University. All Rights Reserved.
#
#  Creator: Hao SUN
#  Description: gym environment, continuous action, cart pole
#  Update Date: 2020-11-10, Hao SUN: renew env para
#  Update Date: 2020-11-13, Hao SUN：add new ddpg demo
#  Update Date: 2020-12-11, Hao SUN：move buffer to trainer
#  Update Date: 2020-12-12, Hao SUN：move create_* files to create_pkg
#  Update Date: 2021-01-01, Hao SUN：chang name


#  General Optimal control Problem Solver (GOPS)


import argparse
from modules.create_pkg.create_env import create_env
from modules.create_pkg.create_alg import create_alg
from modules.create_pkg.create_trainer import create_trainer

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    # Key Parameters for users
    parser.add_argument('--env_id', type=str, default='gym_cartpole_conti', help='')
    parser.add_argument('--apprfunc', type=str, default='MLP', help='')
    parser.add_argument('--algorithm', type=str, default='DDPG', help='')
    parser.add_argument('--trainer', type=str, default='serial_trainer', help='')
    parser.add_argument('--savefile', type=str, default=None, help='')

    # 1. Parameters for environment
    parser.add_argument('--obsv_dim', type=int, default=None, help='')
    parser.add_argument('--action_dim', type=int, default=None, help='')
    parser.add_argument('--action_high_limit', type=list, default=None, help='')
    parser.add_argument('--action_low_limit', type=list, default=None, help='')
    parser.add_argument('--max_episode_length', type=int, default=500, help='')

    # 2. Parameters for approximate function
    parser.add_argument('--value_func_name', type=str, default='critic_q', help='')
    parser.add_argument('--value_func_type', type=str, default=parser.parse_args().apprfunc, help='')
    parser.add_argument('--value_hidden_sizes', type=list, default=[256, 256])
    parser.add_argument('--value_hidden_activation', type=str, default='relu', help='')
    parser.add_argument('--value_output_activation', type=str, default='linear', help='')

    parser.add_argument('--policy_func_name', type=str, default='actor', help='')
    parser.add_argument('--policy_func_type', type=str, default=parser.parse_args().apprfunc, help='')
    parser.add_argument('--policy_hidden_sizes', type=list, default=[256, 256])
    parser.add_argument('--policy_hidden_activation', type=str, default='relu', help='')
    parser.add_argument('--policy_output_activation', type=str, default='tanh', help='')

    # 3. Parameters for algorithm
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005, help='')
    parser.add_argument('--value_learning_rate', type=float, default=1e-3, help='')
    parser.add_argument('--policy_learning_rate', type=float, default=1e-3, help='')
    parser.add_argument('--delay_update', type=int, default=1, help='')
    parser.add_argument('--max_sampled_number', type=int, default=2000)

    # 4. Parameters for trainer
    parser.add_argument('--max_train_episode', type=int, default=2000, help='')
    parser.add_argument('--episode_length', type=int, default=200, help='')
    parser.add_argument('--max_sample_num', type=int, default=100000, help='')
    0
    parser.add_argument('--eval_length', type=int, default=parser.parse_args().cost_horizon, help='')
    parser.add_argument('--buffer_name', type=str, default='replay_buffer')
    parser.add_argument('--buffer_warm_size', type=int, default=1000)
    parser.add_argument('--buffer_max_size', type=int, default=100000)
    parser.add_argument('--noise', type=float, default=0.5, help='')
    parser.add_argument('--reward_scale', type=float, default=0.1, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--is_render', type=bool, default=True)

    # Data savings
    parser.add_argument('--save_folder', type=str,default='./results/' + parser.parse_args().algorithm)
    parser.add_argument('--apprfunc_save_interval', type=int, default=1000)
    parser.add_argument('--log_save_interval', type=int, default=10)

    # get parameter dict
    args = vars(parser.parse_args())

    # Step 1: create environment
    env = create_env(**args)  #
    args['obsv_dim'] = env.observation_space.shape[0]
    args['action_dim'] = env.action_space.shape[0]
    args['action_high_limit'] = env.action_space.high  # NOTE: [type is np.ndarray, not list]
    args['action_low_limit'] = env.action_space.low

    # Step 2: create algorithm and approximate function
    alg = create_alg(**args)  # create appr_model in algo **vars(args)

    # Step 3: create trainer # create buffer in trainer
    trainer = create_trainer(env, alg,**args)

    # start training
    trainer.train()
    print("Training is Done!")

    # start evaluating
    # TODO add eval func

    # save data
    # TODO save all data
    # TODO: save parse to json
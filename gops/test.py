import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env_id', type=str, default='gym_cartpole_conti', help='')
parser.add_argument('--apprfunc', type=str, default='MLP', help='')
parser.add_argument('--algorithm', type=str, default='DDPG', help='')
args = vars(parser.parse_args())
print(args['env_id'])
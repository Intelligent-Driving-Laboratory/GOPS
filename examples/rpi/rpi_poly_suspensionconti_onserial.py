#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Li Jie


import argparse
import os
import math
import numpy as np

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    ################################################
    # Key Parameters for users
    parser.add_argument('--env_id', type=str, default='pyth_suspensionconti', help='')
    parser.add_argument('--algorithm', type=str, default='RPI', help='')
    parser.add_argument('--enable_cuda', default=False, help='Disable CUDA')

    ################################################
    # 1. Parameters for environment
    parser.add_argument('--action_type', type=str, default='continu', help='')
    parser.add_argument('--is_render', type=bool, default=False, help='')
    parser.add_argument('--is_adversary', type=bool, default=True, help='Adversary training')
    parser.add_argument('--is_constrained', type=bool, default=False, help='Constrained training')

    ################################################
    # 2.1 Parameters of value approximate function
    # Options: StateValue/ActionValue/ActionValueDis
    parser.add_argument('--value_func_name', type=str, default='StateValue')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--value_func_type', type=str, default='POLY')
    parser.add_argument('--value_degree', type=int, default=2)
    parser.add_argument('--value_add_bias', type=bool, default=True)
    parser.add_argument('--gt_weight', type=list, default=None)

    # 2.2 Parameters of policy approximate function
    # Options: None/DetermPolicy/StochaPolicy
    parser.add_argument('--policy_func_name', type=str, default='DetermPolicy')
    # Options: MLP/CNN/RNN/POLY/GAUSS
    parser.add_argument('--policy_func_type', type=str, default='POLY')
    parser.add_argument("--policy_act_distribution", type=str, default="default")
    parser.add_argument('--policy_degree', type=int, default=1)
    parser.add_argument('--policy_add_bias', type=bool, default=True)

    ################################################
    # 3. Parameters for algorithm
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--gamma_atte', type=float, default=30)
    parser.add_argument('--norm_matrix', type=list, default=[10, 1, 10, 0.5])
    parser.add_argument('--state_weight', type=list, default=[1000.0, 3.0, 100.0, 0.1])
    parser.add_argument('--control_weight', type=list, default=[1.0])

    ################################################
    # 4. Parameters for trainer
    # Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer
    parser.add_argument('--trainer', type=str, default='on_serial_trainer')
    # Maximum iteration number
    parser.add_argument('--max_newton_iteration', type=int, default=50)
    parser.add_argument('--max_step_update_value', type=int, default=10000)
    parser.add_argument('--max_iteration', type=int, default=parser.parse_args().max_newton_iteration)
    parser.add_argument('--ini_network_dir', type=str, default=None)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument('--sampler_name', type=str, default='on_sampler')
    # Batch size of sampler for buffer store
    parser.add_argument('--sample_batch_size', type=int, default=64, help='Batch size of sampler for buffer store = 64')
    # Add noise to actions for better exploration
    parser.add_argument('--noise_params', type=dict, default=None, help='add noise to actions for exploration')
    parser.add_argument('--probing_noise', type=bool, default=True, help='the persistency of excitation (PE) condition')
    parser.add_argument('--prob_intensity', type=float, default=1.0, help='the intensity of probing noise')
    parser.add_argument('--base_decline', type=float, default=0.0, help='the intensity of probing noise')
    # Initial state
    parser.add_argument('--fixed_initial_state', type=list, default=[0, 0, 0, 0], help='for env_data')
    parser.add_argument('--initial_state_range', type=list, default=[0.05, 0.5, 0.05, 1.0], help='for env_model')
    # State threshold
    parser.add_argument('--state_threshold', type=list, default=[0.08, 0.8, 0.1, 1.6])
    # Rollout steps
    parser.add_argument('--lower_step', type=int, default=200, help='for env_model')
    parser.add_argument('--upper_step', type=int, default=500, help='for env_model')  # shorter, faster but more error
    parser.add_argument('--max_episode_steps', type=int, default=1500, help='for env_data')

    ################################################
    # 6. Parameters for buffer
    parser.add_argument('--buffer_name', type=str, default='replay_buffer')
    parser.add_argument('--buffer_warm_size', type=int, default=parser.parse_args().sample_batch_size)
    parser.add_argument('--buffer_max_size', type=int, default=parser.parse_args().sample_batch_size)

    ################################################
    # 7. Parameters for evaluator
    parser.add_argument('--evaluator_name', type=str, default='evaluator')
    parser.add_argument('--num_eval_episode', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=500000)
    parser.add_argument('--print_interval', type=int, default=1)

    ################################################
    # 8. Data savings
    parser.add_argument('--save_folder', type=str, default=None)
    # Save value/policy every N updates
    parser.add_argument('--apprfunc_save_interval', type=int, default=10, help='Save value/policy every N updates')
    # Save key info every N updates
    parser.add_argument('--log_save_interval', type=int, default=1, help='Save data every N updates')

    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**args)
    args = init_args(env, **args)

    # start_tensorboard(args['save_folder'])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # alg.set_parameters({'gamma': 0.995, 'loss_coefficient_value': 0.5, 'loss_coefficient_entropy': 0.01})
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    # Start training ... ...
    trainer.train()
    print('Training is finished!')

    # # Plot and save training figures
    # plot_all(args['save_folder'])
    # save_tb_to_csv(args['save_folder'])

    # import torch
    # from torch.nn.parameter import Parameter
    # from modules.create_pkg.create_simulator import create_simulator
    #
    # # Create evaluator/simulator in trainer
    # args.update(dict(simulator_name='simulator',
    #                  simulation_step=1501))
    # simulator = create_simulator(**args)
    # simulator.networks.value_target.v.weight = Parameter(
    #     torch.tensor(trainer.value_weight[-1, :], dtype=torch.float32), requires_grad=True)
    # pos_body_tpi = []
    # control_tpi = []
    # attenuation_tpi = []
    # pos_body_without = []
    # control_without = []
    # attenuation_without = []
    #
    # sim_dict_list_tpi = simulator.run_an_episode(args['max_iteration'], dist=True)
    # obs_tpi = np.stack(sim_dict_list_tpi['obs_list'], axis=0)
    # act_tpi = np.stack(sim_dict_list_tpi['action_list'], axis=0)
    # l2_gain_tpi = np.stack(sim_dict_list_tpi['l2_gain_list'], axis=0)
    # pos_body_tpi.append(obs_tpi[:, 0])
    # control_tpi.append(act_tpi[:, 0])
    # attenuation_tpi.append(l2_gain_tpi[:, 0])
    #
    # sim_dict_list_without = simulator.run_an_episode(args['max_iteration'], without_control=True, dist=True)
    # obs_without = np.stack(sim_dict_list_without['obs_list'], axis=0)
    # act_without = np.stack(sim_dict_list_without['action_list'], axis=0)
    # l2_gain_without = np.stack(sim_dict_list_without['l2_gain_list'], axis=0)
    # pos_body_without.append(obs_without[:, 0])
    # control_without.append(act_without[:, 0])
    # attenuation_without.append(l2_gain_without[:, 0])
    #
    # time = np.stack(sim_dict_list_without['time_list'], axis=0)
    # label_list = ['tpi', 'without control']
    # num_data = obs_without.shape[0] - 1
    #
    # save_data(data=obs_tpi, row=num_data + 1, column=args['obsv_dim'],
    #           save_file=args['save_folder'], xls_name='/comp_state_{:d}'.format(num_data))
    # my_plot(data=obs_tpi, time=time,
    #         figure_size_scalar=1,
    #         color_list=None, label_list=['pos_body [m]', 'vel_body [m/s]', 'pos_wheel[m]', 'vel_wheel[m/s]'],
    #         loc_legend='upper right', ncol=1,
    #         xlim=(0, time[-1, 0]), ylim=None,
    #         xtick=None, ytick=None,
    #         xlabel='time [s]', ylabel='state',
    #         xline=None, yline=None,
    #         pad=None,
    #         figure_name=args['save_folder'] + '/comp_state_{:d}'.format(num_data), figure_type='png',
    #         display=False)
    #
    # save_data(data=np.stack(pos_body_tpi + pos_body_without, axis=1), row=num_data + 1, column=len(label_list),
    #           save_file=args['save_folder'], xls_name='/comp_pos_body_{:d}'.format(num_data))
    # my_plot(data=np.stack(pos_body_tpi + pos_body_without, axis=1), time=time,
    #         figure_size_scalar=1,
    #         color_list=None, label_list=label_list,
    #         loc_legend='upper right', ncol=1,
    #         xlim=(0, time[-1, 0]), ylim=None,
    #         xtick=None, ytick=None,
    #         xlabel='time [s]', ylabel='the positions of the car body [m]',
    #         xline=None, yline=None,
    #         pad=None,
    #         figure_name=args['save_folder'] + '/comp_pos_body_{:d}'.format(num_data), figure_type='png',
    #         display=False)
    #
    # save_data(data=np.stack(control_tpi + control_without, axis=1), row=num_data + 1, column=len(label_list),
    #           save_file=args['save_folder'], xls_name='/comp_control_force_{:d}'.format(num_data))
    # my_plot(data=np.stack(control_tpi + control_without, axis=1), time=time,
    #         figure_size_scalar=1,
    #         color_list=None, label_list=label_list,
    #         loc_legend='lower right', ncol=1,
    #         xlim=(0, time[-1, 0]), ylim=None,
    #         xtick=None, ytick=None,
    #         xlabel='time [s]', ylabel='control force [kN]',
    #         xline=None, yline=None,
    #         pad=None,
    #         figure_name=args['save_folder'] + '/comp_control_force_{:d}'.format(num_data), figure_type='png',
    #         display=False)
    #
    # save_data(data=np.stack(attenuation_tpi + attenuation_without, axis=1), row=num_data + 1, column=len(label_list),
    #           save_file=args['save_folder'], xls_name='/comp_attenuation_{:d}'.format(num_data))
    # my_plot(data=np.stack(attenuation_tpi + attenuation_without, axis=1), time=time,
    #         figure_size_scalar=1,
    #         color_list=None, label_list=label_list,
    #         loc_legend='lower right', ncol=1,
    #         xlim=(0, time[-1, 0]), ylim=None,
    #         xtick=None, ytick=None,
    #         xlabel='time [s]', ylabel='attenuation',
    #         xline=None, yline=None,
    #         pad=None,
    #         figure_name=args['save_folder'] + '/comp_attenuation_{:d}'.format(num_data), figure_type='png',
    #         display=False)
    #
    # data_value_weight = trainer.value_weight
    # num_data = data_value_weight.shape[0] - 1
    # num_line = data_value_weight.shape[1]
    # # gt = np.array([[2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # # gt_value_weight = gt.repeat(num_data + 1, axis=0)
    # save_data(data=data_value_weight, row=num_data + 1, column=num_line,
    #           save_file=args['save_folder'], xls_name='/value_weight_{:d}'.format(num_data))
    # my_plot(data=data_value_weight, gt=None,
    #         figure_size_scalar=1,
    #         color_list=None, label_list=[r'$\mathregular{{\omega}_{' + str(i + 1) + '}}$' for i in range(num_line)],
    #         loc_legend='center right', ncol=1, style_legend='italic', font_size_legend=9,
    #         xlim=(0, num_data), ylim=None,
    #         xtick=None, ytick=None,
    #         xlabel='iteration', ylabel='weights of value network',
    #         xline=None, yline=None,
    #         pad=None,
    #         figure_name=args['save_folder'] + '/value_weight_{:d}'.format(num_data), figure_type='png',
    #         display=True)

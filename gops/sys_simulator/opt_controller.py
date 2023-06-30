#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Implementation of optimal controller based on MPC
#  Update: 2022-12-05, Zhilong Zheng: create OptController


import time
from typing import Callable, List, Optional, Tuple, Union
import warnings
from gops.env.env_gen_ocp.pyth_base import State
from gops.env.env_gen_ocp.env_model.pyth_model_base import EnvModel
import torch
from functorch import jacrev
import numpy as np
from cyipopt import minimize_ipopt
import scipy.optimize as opt
from mytimeit import timeit, Timeit


MUTE = False

class OptController:
    """Implementation of optimal controller based on MPC.

    :param EnvModel model: model of environment to work on
    :param int num_pred_step: Total steps of prediction, specifying how far to look into future.
    :param int ctrl_interval:
        (Optional) Optimal control inputs are computed every `ctrl_interval` steps, **it should be factor of num_pred_step**.
        For example, if `num_pred_step` equals 10, and `ctrl_interval` equals 2,
        then control inputs will be computed at timestep 0, 2, 4, 6 and 8.
        Control inputs at rest timesteps are set in zero-order holder manner. Default to 1.
    :param float gamma: (Optional) Discounting factor. Valid range: [0, 1]. Default to 1.0.
    :param bool use_terminal_cost: (Optional) Whether to use terminal cost. Default to False.
    :param Callable[[torch.Tensor], torch.Tensor] terminal_cost:
        (Optional) Self-defined terminal cost function returning one Tensor of shape [] (scalar).
        If `use_terminal_cost` is True and `terminal_cost` is None,
        OptController will use default terminal cost function of environment model (if exists). Default to None.
    :param dict minimize_options:
        (Optional) Options for minimizing to be passed to IPOPT.
        See [IPOPT Options](https://coin-or.github.io/Ipopt/OPTIONS.html) for details. Default to None.
    :param int verbose: (Optional) Whether to print summary statistics. Valid value: {0, 1}. Default to 0.
    :param str mode:
        (Optional) Specify method to be used to solve optimal control problem.
        Valid value: {"shooting", "collocation"}. Default to "collocation".
    """

    def __init__(
        self,
        model: EnvModel,
        num_pred_step: int,
        ctrl_interval: int = 1,
        gamma: float = 1.0,
        use_terminal_cost: bool = False,
        terminal_cost: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        minimize_options: Optional[dict] = None,
        verbose: int = 0,
        mode: str = "collocation",
    ):

        self.model = model
        self.sim_dt = model.dt
        self.state_dim = model.robot_model.robot_state_dim
        self.action_dim = model.action_dim
        self.StateClass = model.StateClass

        self.gamma = gamma

        self.ctrl_interval = ctrl_interval
        self.num_pred_step = num_pred_step
        assert (
            num_pred_step % ctrl_interval == 0
        ), "ctrl_interval should be a factor of num_pred_step."
        self.num_ctrl_points = int(num_pred_step / ctrl_interval)

        assert mode in ["shooting", "collocation"]
        self.mode = mode
        if self.mode == "shooting":
            self.rollout_mode = "loop"
        elif self.mode == "collocation":
            self.rollout_mode = "batch"

        if use_terminal_cost:
            if terminal_cost is not None:
                self.terminal_cost = terminal_cost
            else:
                self.terminal_cost = model.get_terminal_cost
            assert (
                self.terminal_cost is not None
            ), "Choose to use terminal cost, but there is no available terminal cost function."
        else:
            if terminal_cost is not None:
                warnings.warn(
                    "Choose not to use terminal cost, but a terminal cost function is given. This will be ignored."
                )
            self.terminal_cost = None

        self.minimize_options = minimize_options
        if self.mode == "shooting":
            lower_bound = self.model.action_lower_bound
            upper_bound = self.model.action_upper_bound
            self.optimize_dim = self.action_dim
        elif self.mode == "collocation":
            lower_bound = torch.cat(
                (self.model.action_lower_bound, self.model.robot_model.robot_state_lower_bound)
            )
            upper_bound = torch.cat(
                (self.model.action_upper_bound, self.model.robot_model.robot_state_upper_bound)
            )
            self.optimize_dim = self.action_dim + self.state_dim
        self.initial_guess = np.zeros(self.optimize_dim * self.num_ctrl_points)
        self.bounds = opt.Bounds(
            np.tile(lower_bound, (self.num_ctrl_points,)),
            np.tile(upper_bound, (self.num_ctrl_points,)),
        )

        self.verbose = verbose
        self._reset_statistics()

    def __call__(self, x: State[np.ndarray]) -> np.ndarray:
        """Compute optimal control input for current state

        :param State x: Current state
        :param InfoDict info: (Optional) Additional info that are required by model. Default to {}.

        Return: optimal control input for current state x
        """
        
        x = self.StateClass.array2tensor(x)
        x = self.StateClass.stack([x])
        # if info:
        #     info = info.copy()
        #     for (key, value) in info.items():
        #         info[key] = torch.tensor(value, dtype=torch.float32)
        res = minimize_ipopt(
            fun=self._cost_fcn_and_jac,
            x0=self.initial_guess,
            args=(x,),
            jac=True,
            bounds=opt._constraints.new_bounds_to_old(
                self.bounds.lb, self.bounds.ub, self.num_ctrl_points * self.optimize_dim
            ),
            constraints=[
                {
                    "type": "ineq",
                    "fun": self._constraint_fcn,
                    "jac": self._constraint_jac,
                    "args": (x,),
                },
                {
                    "type": "eq",
                    "fun": self._trans_constraint_fcn,
                    "jac": self._trans_constraint_jac,
                    "args": (x,),
                },
            ],
            options=self.minimize_options,
        )
        self.initial_guess = np.concatenate(
            (res.x[self.optimize_dim :], res.x[-self.optimize_dim :])
        )
        if self.verbose > 0:
            self._print_statistics(res)
        return res.x.reshape((self.num_ctrl_points, self.optimize_dim))[
            0, : self.action_dim
        ]

    def _cost_fcn_and_jac(
        self, inputs: np.ndarray, x: State[torch.Tensor]
    ) -> Tuple[float, np.ndarray]:
        """
        Compute value and jacobian of cost function
        """
        inputs = self._preprocess_inputs(inputs, requires_grad=True)
        cost = self._compute_cost(inputs, x)
        jac = torch.autograd.grad(cost, inputs)[0]
        return cost.detach().item(), jac.numpy().astype("d")

    def _constraint_fcn(
        self, inputs: Union[np.ndarray, torch.Tensor], x: State[torch.Tensor]
    ) -> torch.Tensor:

        if self.model.get_constraint is None:
            return torch.tensor([0.0])
        else:
            self.constraint_evaluations += 1
            inputs = self._preprocess_inputs(inputs)
            states = self._rollout(inputs, x)

            # model.get_constraint() returns Tensor, each element of which
            # should be required to be lower than or equal to 0
            # minimize_ipopt() takes inequality constraints that should be greater than or equal to 0
            cstr_vector = -self.model.get_constraint(self.StateClass.concat(states)).reshape(-1)
            return cstr_vector

    def _constraint_jac(
        self, inputs: np.ndarray, x: State[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute jacobian of constraint function
        """
        inputs = torch.tensor(inputs, dtype=torch.float32)
        jac = jacrev(self._constraint_fcn)(inputs, x)
        return jac.numpy().astype("d")

    def _trans_constraint_fcn(
        self, inputs: Union[np.ndarray, torch.Tensor], x: State[torch.Tensor]
    ) -> torch.Tensor:
        """
        Transition constraint function (collocation only)
        """
        if self.mode == "shooting":
            return torch.tensor([0.0])
        elif self.mode == "collocation":
            self.constraint_evaluations += 1

            inputs = self._preprocess_inputs(inputs)
            true_states = self._rollout(inputs, x)
            true_states = self.StateClass.concat(true_states).robot_state
            true_states = true_states[1 :: self.ctrl_interval, :].reshape(-1)
            input_states = inputs.reshape((-1, self.optimize_dim))[
                :, -self.state_dim :
            ].reshape(-1)
            return true_states - input_states

    def _trans_constraint_jac(
        self, inputs: np.ndarray, x: State[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute jacobian of transition constraint function (collocation only)
        """
        inputs = torch.tensor(inputs, dtype=torch.float32)
        jac = jacrev(self._trans_constraint_fcn)(inputs, x)
        return jac.numpy().astype("d")
    
    @timeit(mute=True)
    def _rollout(
        self, inputs: torch.Tensor, x: State[torch.Tensor]
    ) -> List[State[torch.Tensor]]:
        """
        Rollout furture states using environment model
        """
        st = time.time()
        self.system_simulations += 1
        states = [x]

        if self.rollout_mode == "loop":
            # next_x = x.unsqueeze(0)
            # batched_info = {}
            # if info:
            #     batched_info = {key: value.unsqueeze(0) for key, value in info.items()}
            #     infos = {key: [value.unsqueeze(0)] for key, value in info.items()}
            for i in np.arange(0, self.num_pred_step):
                u = inputs[i, : self.action_dim].unsqueeze(0)
                states.append(self.model.get_next_state(states[-1], u))
                # rewards[i] = -reward * (self.gamma ** i)
                
                # if info:
                #     for key in info.keys():
                #         infos[key].append(batched_info[key])
            
        elif self.rollout_mode == "batch":
            with Timeit("main_rollout", mute=True):
                with Timeit("robot state rollout", mute=False):
                    # rollout robot states batch-wise
                    robot_states = torch.zeros((self.num_pred_step + 1, self.state_dim))
                    robot_states[0, :] = x.robot_state
                    xs = torch.cat(
                        (
                            x.robot_state,
                            inputs[
                                : -self.ctrl_interval : self.ctrl_interval,
                                -self.state_dim :,
                            ],
                        )
                    )
                    us = inputs[:: self.ctrl_interval, : self.action_dim]
                    for i in range(self.ctrl_interval):
                        xs = self.model.robot_model.get_next_state(xs, us)
                        robot_states[i + 1 :: self.ctrl_interval, :] = xs
                
                with Timeit("context_rollout", mute=False):
                    # rollout context states
                    context_state = x.context_state
                    for i in np.arange(0, self.num_pred_step):
                        context_state = self.model.context_model.get_next_state(
                            context_state, 
                            inputs[i:i + 1, : self.action_dim]
                        )
                        states.append(self.StateClass(robot_states[i + 1:i + 2, :], context_state))
        
        et = time.time()
        self.system_simulation_time += et - st
        return states

    # @timeit(mute=MUTE)
    def _compute_cost(
        self, inputs: torch.Tensor, x: State[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total cost of optimization inputs
        """
        # rollout states and rewards
        states = self._rollout(inputs, x)
        rewards = self.model.get_reward(self.StateClass.concat(states[:-1]), inputs[:, : self.action_dim])
        # sum up integral costs from timestep 0 to T-1
        cost = -rewards @ torch.logspace(
            0, self.num_pred_step - 1, self.num_pred_step, base=self.gamma
        )

        # Terminal cost for timestep T
        if self.terminal_cost is not None:
            terminal_cost = self.terminal_cost(states[-1, :])
            cost += terminal_cost * (self.gamma ** self.num_pred_step)
        return cost

    def _preprocess_inputs(
        self, 
        inputs: Union[np.ndarray, torch.Tensor], 
        requires_grad=False
    ) -> torch.Tensor:
        """
        Preprocess inputs to fit into environment model
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=requires_grad)
        inputs = inputs.reshape((self.num_ctrl_points, self.optimize_dim))
        if self.ctrl_interval > 1:
            inputs = inputs.repeat_interleave(self.ctrl_interval, dim=0)  #TODO: speed up this
        return inputs

    def _reset_statistics(self):
        """
        Reset counters for keeping track of statistics
        """
        self.constraint_evaluations = 0
        self.system_simulations = 0
        self.system_simulation_time = 0

    def _print_statistics(self, res: opt.OptimizeResult, reset=True):
        """
        Print out summary statistics from last run
        """
        print(res.message)
        print("Summary statistics:")
        print("* Number of iterations:", res.nit)
        print("* Cost function calls:", res.nfev)
        if self.constraint_evaluations:
            print("* Constraint calls:", self.constraint_evaluations)
        print("* System simulations:", self.system_simulations)
        print("* System simulation time:", self.system_simulation_time)
        print("* Final cost:", res.fun, "\n")
        if reset:
            self._reset_statistics()



if __name__ == "__main__":
    import argparse
    import time
    from gops.create_pkg.create_env import create_env
    from gops.create_pkg.create_env_model import create_env_model
    import matplotlib.pyplot as plt
    import imageio
    import os
    # Parameters Setup
    parser = argparse.ArgumentParser()
    env_id = "pyth_veh2dofconti_errcstr"
    parser.add_argument("--env_id", type=str, default=env_id)
    parser.add_argument("--lq_config", type=str, default="s6a3")
    parser.add_argument('--clip_action', type=bool, default=True)
    parser.add_argument('--clip_obs', type=bool, default=False)
    parser.add_argument('--mask_at_done', type=bool, default=False)
    parser.add_argument(
        "--is_adversary", type=bool, default=False, help="Adversary training"
    )
    parser.add_argument('--sample_batch_size', type=int, default=64, help='Batch size of sampler for buffer store = 64')

    if env_id == "pyth_aircraftconti":
        parser.add_argument('--max_episode_steps', type=int, default=200)
        parser.add_argument('--gamma_atte', type=float, default=5)
        parser.add_argument('--fixed_initial_state', type=list, default=[1.0, 1.5, 1.0], help='for env_data')
        parser.add_argument('--initial_state_range', type=list, default=[0.1, 0.2, 0.1], help='for env_model')
        parser.add_argument('--state_threshold', type=list, default=[2.0, 2.0, 2.0])
        parser.add_argument('--lower_step', type=int, default=200, help='for env_model')
        parser.add_argument('--upper_step', type=int, default=700, help='for env_model')
    
    if env_id == "pyth_oscillatorconti":
        parser.add_argument('--max_episode_steps', type=int, default=200)
        parser.add_argument('--gamma_atte', type=float, default=2)
        parser.add_argument('--fixed_initial_state', type=list, default=[0.5, -0.5], help='for env_data [0.5, -0.5]')
        parser.add_argument('--initial_state_range', type=list, default=[1.5, 1.5], help='for env_model')
        parser.add_argument('--state_threshold', type=list, default=[5.0, 5.0])
        parser.add_argument('--lower_step', type=int, default=200, help='for env_model')
        parser.add_argument('--upper_step', type=int, default=700, help='for env_model')

    if env_id == "pyth_suspensionconti":
        parser.add_argument('--gamma_atte', type=float, default=30)
        parser.add_argument('--state_weight', type=list, default=[1000.0, 3.0, 100.0, 0.1])
        parser.add_argument('--control_weight', type=list, default=[1.0])
        parser.add_argument('--fixed_initial_state', type=list, default=[0, 0, 0, 0], help='for env_data')
        parser.add_argument('--initial_state_range', type=list, default=[0.05, 0.5, 0.05, 1.0], help='for env_model')
        # State threshold
        parser.add_argument('--state_threshold', type=list, default=[0.08, 0.8, 0.1, 1.6])
        parser.add_argument('--lower_step', type=int, default=200, help='for env_model')
        parser.add_argument('--upper_step', type=int, default=500, help='for env_model')  # shorter, faster but more error
        parser.add_argument('--max_episode_steps', type=int, default=1500, help='for env_data')
        parser.add_argument('--max_newton_iteration', type=int, default=50)
        parser.add_argument('--max_iteration', type=int, default=parser.parse_args().max_newton_iteration)
    
    if env_id == "pyth_veh2dofconti_errcstr" or env_id == "pyth_veh3dofconti_errcstr":
        parser.add_argument("--pre_horizon", type=int, default=5)
    
    if env_id == "pyth_veh3dofconti_surrcstr":
        parser.add_argument("--pre_horizon", type=int, default=10)
        parser.add_argument("--surr_veh_num", type=int, default=4)

    args = vars(parser.parse_args())
    env_model = create_env_model(**args)
    state_dim = env_model.state_dim
    if env_id == "pyth_veh2dofconti_errcstr":
        state_dim -= args["pre_horizon"]
    elif env_id == "pyth_veh3dofconti_errcstr":
        state_dim -= 2 * args["pre_horizon"]
    elif env_id == "pyth_veh3dofconti_surrcstr":
        state_dim = env_model.state_dim
    action_dim = env_model.action_dim

    if args["env_id"] == "pyth_lq":
        max_state_errs = []
        mean_state_errs = []
        max_action_errs = []
        mean_action_errs = []
        K = env_model.dynamics.K
    
    times = []
    seed = 3
    # num_pred_steps = range(10, 110, 20)
    num_pred_steps = (5,)
    ctrl_interval = 1
    sim_num = 50
    sim_horizon = np.arange(sim_num)
    for num_pred_step in num_pred_steps:
        controller = OptController(
            env_model, 
            ctrl_interval=ctrl_interval, 
            num_pred_step=num_pred_step, 
            gamma=0.99,
            verbose=1,
            minimize_options={
                "max_iter": 200, 
                "tol": 1e-3,
                "acceptable_tol": 1e-1,
                "acceptable_iter": 10,
                # "print_level": 3,
                # "print_timing_statistics": "yes",
            },
            # use_terminal_cost=True,
            mode="shooting",
        )

        env = create_env(**args)
        env.seed(seed)
        x, info = env.reset()
        # env.render()
        filenames = []
        xs = []
        us= []
        rs = []
        ts = []
        for i in sim_horizon:
            print(f"step: {i + 1}")
            if (i % ctrl_interval) == 0:
                t1 = time.time()
                u = controller(x.astype(np.float32), info)
                t2 = time.time()
                print(f"total time: {t2 - t1}")
            xs.append(x)
            us.append(u)
            ts.append(t2 - t1)
            x, r, _, info = env.step(u)
            rs.append(r)
            # env.render("rgb_array")
            # filename = f'{i}.png'
            # filenames.append(filename)
            # plt.savefig(filename)
            # print(f"x:{x[:6]}")
            # print(f"u:{u}\n")
        xs = np.stack(xs)
        us = np.stack(us)
        rs = np.stack(rs)
        times.append(ts)

        if args["env_id"] == "pyth_lq":
            env.seed(seed)
            x, _ = env.reset()
            xs_lqr = []
            us_lqr= []
            for i in sim_horizon:
                u = -K @ x
                xs_lqr.append(x)
                us_lqr.append(u)
                x, _, _, _ = env.step(u)
            xs_lqr = np.stack(xs_lqr)
            us_lqr = np.stack(us_lqr)

            max_state_err = np.max(np.abs(xs - xs_lqr), axis=0) / (np.max(xs_lqr, axis=0) - np.min(xs_lqr, axis=0)) * 100
            mean_state_err = np.mean(np.abs(xs - xs_lqr), axis=0) / (np.max(xs_lqr, axis=0) - np.min(xs_lqr, axis=0)) * 100
            max_action_err = np.max(np.abs(us - us_lqr), axis=0) / (np.max(us_lqr, axis=0) - np.min(us_lqr, axis=0)) * 100
            mean_action_err = np.mean(np.abs(us - us_lqr), axis=0) / (np.max(us_lqr, axis=0) - np.min(us_lqr, axis=0)) * 100
            max_state_errs.append(max_state_err)
            mean_state_errs.append(mean_state_err)
            max_action_errs.append(max_action_err)
            mean_action_errs.append(mean_action_err)

    if args["env_id"] == "pyth_lq":
        max_state_errs = np.stack(max_state_errs)
        mean_state_errs = np.stack(mean_state_errs)
        max_action_errs = np.stack(max_action_errs)
        mean_action_errs = np.stack(mean_action_errs)

    #=======state-timestep=======#
    plt.figure()
    for i in range(state_dim):
        plt.subplot(state_dim, 1, i + 1)
        plt.plot(sim_horizon, xs[:, i], label="mpc")
        if args["env_id"] == "pyth_lq":
            plt.plot(sim_horizon, xs_lqr[:, i], label="lqr")
            print(f"State-{i+1} Max error: {round(max_state_err[i], 3)}%, Mean error: {round(mean_state_err[i], 3)}%")
        plt.ylabel(f"State-{i+1}")
    plt.legend()
    plt.xlabel("Time Step")
    if args["env_id"] == "pyth_lq":
        plt.savefig(f"State-{args['env_id']}-{args['lq_config']}.png")
    else:
        plt.savefig(f"State-{args['env_id']}.png")

    #=======action-timestep=======#
    plt.figure()
    for i in range(action_dim):
        plt.subplot(action_dim, 1, i + 1)
        plt.plot(sim_horizon[::ctrl_interval], us[::ctrl_interval, i], label="mpc")
        if args["env_id"] == "pyth_lq":
            plt.plot(sim_horizon, us_lqr[:, i], label="lqr")
            print(f"Action-{i+1} Max error: {round(max_action_err[i], 3)}%, Mean error: {round(mean_action_err[i], 3)}%")
        plt.ylabel(f"Action-{i+1}")
    plt.legend()
    plt.xlabel("Time Step")
    if args["env_id"] == "pyth_lq":
        plt.savefig(f"Action-{args['env_id']}-{args['lq_config']}.png")
    else:
        plt.savefig(f"Action-{args['env_id']}.png")

    #=======reward-timestep=======#
    plt.figure()
    plt.plot(sim_horizon, rs, label="mpc")
    plt.ylabel(f"Reward")
    plt.legend()
    plt.xlabel("Time Step")
    if args["env_id"] == "pyth_lq":
        plt.savefig(f"Reward-{args['env_id']}-{args['lq_config']}.png")
    else:
        plt.savefig(f"Reward-{args['env_id']}.png")
    plt.close()

    #=======MPC solving times=======#
    plt.figure()
    plt.boxplot(times, labels=num_pred_steps, showfliers=False)
    plt.xlabel("num pred step")
    plt.ylabel(f"Time (s)")
    if args["env_id"] == "pyth_lq":
        plt.savefig(f"MPC-solving-time-{args['env_id']}-{args['lq_config']}.png")
    else:
        plt.savefig(f"MPC-solving-time-{args['env_id']}.png")
    plt.close()

    #=======error-predstep=======#
    if args["env_id"] == "pyth_lq":
        plt.figure()
        for i in range(state_dim):
            plt.plot(num_pred_steps, max_state_errs[:, i], label=f"State-{i+1}")
        plt.legend()
        plt.xlabel("num pred step")
        plt.ylabel(f"State Max error (%)")
        plt.yscale ('log')
        plt.savefig(f"State-max-err-{args['env_id']}-{args['lq_config']}.png")
        plt.close()

        plt.figure()
        for i in range(action_dim):
            plt.plot(num_pred_steps, max_action_errs[:, i], label=f"Action-{i+1}")
        plt.legend()
        plt.xlabel("num pred step")
        plt.ylabel(f"Action Max error (%)")
        plt.yscale ('log')
        plt.savefig(f"Action-max-err-{args['env_id']}-{args['lq_config']}.png")
        plt.close()

        plt.figure()
        for i in range(state_dim):
            plt.plot(num_pred_steps, mean_state_errs[:, i], label=f"State-{i+1}")
        plt.legend()
        plt.xlabel("num pred step")
        plt.ylabel(f"State Mean error (%)")
        plt.yscale ('log')
        plt.savefig(f"State-mean-err-{args['env_id']}-{args['lq_config']}.png")
        plt.close()

        plt.figure()
        for i in range(action_dim):
            plt.plot(num_pred_steps, mean_action_errs[:, i], label=f"Action-{i+1}")
        plt.legend()
        plt.xlabel("num pred step")
        plt.ylabel(f"Action Mean error (%)")
        plt.yscale ('log')
        plt.savefig(f"Action-mean-err-{args['env_id']}-{args['lq_config']}.png")
        plt.close()

    with imageio.get_writer('mygif.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    for filename in set(filenames):
        os.remove(filename)
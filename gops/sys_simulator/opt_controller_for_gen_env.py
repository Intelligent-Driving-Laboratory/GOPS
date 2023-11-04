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
import warnings
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import scipy.optimize as opt
import torch
from cyipopt import minimize_ipopt
from functorch import jacrev
from gops.env.env_gen_ocp.env_model.pyth_base_model import EnvModel
from gops.env.env_gen_ocp.pyth_base import State, batch_context_state


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
        
        x = x.array2tensor()

        constraints = []
        if self.model.get_constraint is not None:
            constraints.append(
                {
                    "type": "ineq",
                    "fun": self._constraint_fcn,
                    "jac": self._constraint_jac,
                    "args": (x,),
                }
            )
        if self.mode == "collocation":
            constraints.append(
                {
                    "type": "eq",
                    "fun": self._trans_constraint_fcn,
                    "jac": self._trans_constraint_jac,
                    "args": (x,),
                }
            )
        
        t1 = time.time()
        res = minimize_ipopt(
            fun=self._cost_fcn_and_jac,
            x0=self.initial_guess,
            args=(x,),
            jac=True,
            bounds=opt._constraints.new_bounds_to_old(
                self.bounds.lb, self.bounds.ub, self.num_ctrl_points * self.optimize_dim
            ),
            constraints=constraints,
            options=self.minimize_options,
        )
        self.initial_guess = np.concatenate(
            (res.x[self.optimize_dim :], res.x[-self.optimize_dim :])
        )
        t2 = time.time()
        self.total_time = t2 - t1
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
        self.constraint_evaluations += 1

        inputs = self._preprocess_inputs(inputs)
        state = self._rollout(inputs, x)

        # model.get_constraint() returns Tensor, each element of which
        # should be required to be lower than or equal to 0
        # minimize_ipopt() takes inequality constraints that should be greater than or equal to 0
        cstr_vector = -self.model.get_constraint(state).reshape(-1)
        return cstr_vector

    def _constraint_jac(
        self, inputs: np.ndarray, x: State[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute jacobian of constraint function
        """
        inputs = torch.tensor(inputs, dtype=torch.float32)
        jac = jacrev(partial(self._constraint_fcn, x=x))(inputs)
        return jac.numpy().astype("d")

    def _trans_constraint_fcn(
        self, inputs: Union[np.ndarray, torch.Tensor], x: State[torch.Tensor]
    ) -> torch.Tensor:
        """
        Transition constraint function (collocation only)
        """
        self.constraint_evaluations += 1

        inputs = self._preprocess_inputs(inputs)
        true_states = self._rollout(inputs, x).robot_state
        true_states = true_states[1 :: self.ctrl_interval, :]
        input_states = inputs[:, -self.state_dim :]
        return (true_states - input_states).reshape(-1)

    def _trans_constraint_jac(
        self, inputs: np.ndarray, x: State[torch.Tensor]
    ) -> np.ndarray:
        """
        Compute jacobian of transition constraint function (collocation only)
        """
        inputs = torch.tensor(inputs, dtype=torch.float32)
        jac = jacrev(partial(self._trans_constraint_fcn, x=x))(inputs)
        return jac.numpy().astype("d")

    def _rollout(
        self, inputs: torch.Tensor, x: State[torch.Tensor]
    ) -> State[torch.Tensor]:
        """
        Rollout furture states using environment model
        """
        st = time.time()
        self.system_simulations += 1

        if self.rollout_mode == "loop":
            states = [x.batch(batch_size=1)]
            for i in np.arange(0, self.num_pred_step):
                u = inputs[i, : self.action_dim].unsqueeze(0)
                states.append(self.model.get_next_state(states[-1], u))
            state = State.concat(states)

        elif self.rollout_mode == "batch":
            # rollout robot states batch-wise
            robot_states = torch.zeros((self.num_pred_step + 1, self.state_dim))
            robot_states[0, :] = x.robot_state
            xs = torch.cat(
                (
                    x.robot_state.unsqueeze(0),
                    inputs[
                        : -self.ctrl_interval : self.ctrl_interval,
                        -self.state_dim :,
                    ],
                )
            )
            us = inputs[:: self.ctrl_interval, : self.action_dim]
            for i in range(self.ctrl_interval):
                xs = self.model.robot_model_get_next_state(xs, us)
                robot_states[i + 1 :: self.ctrl_interval, :] = xs
            
            context_states = batch_context_state(x.context_state, self.num_pred_step + 1)
            state = State(robot_states, context_states)
        
        state.context_state.t = torch.arange(0, self.num_pred_step + 1)
        
        et = time.time()
        self.system_simulation_time += et - st
        return state

    def _compute_cost(
        self, inputs: torch.Tensor, x: State[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute total cost of optimization inputs
        """
        # rollout states and rewards
        state = self._rollout(inputs, x)
        rewards = self.model.get_reward(state[:-1], inputs[:, : self.action_dim])
        # sum up integral costs from timestep 0 to T-1
        cost = -rewards @ torch.logspace(
            0, self.num_pred_step - 1, self.num_pred_step, base=self.gamma
        )

        # Terminal cost for timestep T
        if self.terminal_cost is not None:
            terminal_cost = self.terminal_cost(state[-1, :])
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
        if res.status == 0:
            print("\033[92m" + str(res.message, encoding='utf-8') + "\033[0m")  # green
        elif res.status == 1:
            print("\033[32m" + str(res.message, encoding='utf-8') + "\033[0m")  # lightgreen
        else:
            print("\033[91m" + str(res.message, encoding='utf-8') + "\033[0m")  # red
        print("Summary statistics:")
        print("* Number of iterations:", res.nit)
        print("* Cost function calls:", res.nfev)
        if self.constraint_evaluations:
            print("* Constraint calls:", self.constraint_evaluations)
        print("* System simulations:", self.system_simulations)
        print("* System simulation time:", self.system_simulation_time)
        print("* Total time:", self.total_time)
        print("* Final cost:", res.fun, "\n")
        if reset:
            self._reset_statistics()

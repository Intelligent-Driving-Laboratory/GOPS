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
from typing import Callable, Optional, Tuple, Union
import warnings
from gops.env.env_ocp.env_model.pyth_base_model import PythBaseModel
from gops.utils.gops_typing import InfoDict
import torch
from functorch import jacrev
import numpy as np
from cyipopt import minimize_ipopt
import scipy.optimize as opt


class OptController:
    """Implementation of optimal controller based on MPC.

    :param PythBaseModel model: model of environment to work on
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
        model: PythBaseModel,
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
        self.obs_dim = model.obs_dim
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
                (self.model.action_lower_bound, self.model.obs_lower_bound)
            )
            upper_bound = torch.cat(
                (self.model.action_upper_bound, self.model.obs_upper_bound)
            )
            self.optimize_dim = self.action_dim + self.obs_dim
        self.initial_guess = np.zeros(self.optimize_dim * self.num_ctrl_points)
        self.bounds = opt.Bounds(
            np.tile(lower_bound, (self.num_ctrl_points,)),
            np.tile(upper_bound, (self.num_ctrl_points,)),
        )

        self.verbose = verbose
        self._reset_statistics()

    def __call__(self, x: np.ndarray, info: InfoDict = {}) -> np.ndarray:
        """Compute optimal control input for current state

        :param np.ndarray x: Current state
        :param InfoDict info: (Optional) Additional info that are required by model. Default to {}.

        Return: optimal control input for current state x
        """
        x = torch.tensor(x, dtype=torch.float32)
        if info:
            info = info.copy()
            for (key, value) in info.items():
                info[key] = torch.tensor(value, dtype=torch.float32)
        res = minimize_ipopt(
            fun=self._cost_fcn_and_jac,
            x0=self.initial_guess,
            args=(x, info),
            jac=True,
            bounds=opt._constraints.new_bounds_to_old(
                self.bounds.lb, self.bounds.ub, self.num_ctrl_points * self.optimize_dim
            ),
            constraints=[
                {
                    "type": "ineq",
                    "fun": self._constraint_fcn,
                    "jac": self._constraint_jac,
                    "args": (x, info),
                },
                {
                    "type": "eq",
                    "fun": self._trans_constraint_fcn,
                    "jac": self._trans_constraint_jac,
                    "args": (x, info),
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
        self, inputs: np.ndarray, x: torch.Tensor, info: InfoDict
    ) -> Tuple[float, np.ndarray]:
        """
        Compute value and jacobian of cost function
        """
        inputs = torch.tensor(inputs, dtype=torch.float32, requires_grad=True)
        cost = self._compute_cost(inputs, x, info)
        jac = torch.autograd.grad(cost, inputs)[0]
        return cost.detach().item(), jac.numpy().astype("d")

    def _constraint_fcn(
        self, inputs: Union[np.ndarray, torch.Tensor], x: torch.Tensor, info: InfoDict
    ) -> torch.Tensor:

        if self.model.get_constraint is None:
            return torch.tensor([0.0])
        else:
            self.constraint_evaluations += 1

            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            states, _, infos = self._rollout(inputs, x, info)
            if info:
                for key in info.keys():
                    infos[key] = torch.cat(infos[key])

            # model.get_constraint() returns Tensor, each element of which
            # should be required to be lower than or equal to 0
            # minimize_ipopt() takes inequality constraints that should be greater than or equal to 0
            cstr_vector = -self.model.get_constraint(states, infos).reshape(-1)
            return cstr_vector

    def _constraint_jac(
        self, inputs: np.ndarray, x: torch.Tensor, info: InfoDict
    ) -> np.ndarray:
        """
        Compute jacobian of constraint function
        """
        inputs = torch.tensor(inputs, dtype=torch.float32)
        jac = jacrev(self._constraint_fcn)(inputs, x, info)
        return jac.numpy().astype("d")

    def _trans_constraint_fcn(
        self, inputs: Union[np.ndarray, torch.Tensor], x: torch.Tensor, info: InfoDict
    ) -> torch.Tensor:
        """
        Transition constraint function (collocation only)
        """
        if self.mode == "shooting":
            return torch.tensor([0.0])
        elif self.mode == "collocation":
            self.constraint_evaluations += 1

            if isinstance(inputs, np.ndarray):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            true_states, _, _ = self._rollout(inputs, x, info)
            true_states = true_states[1 :: self.ctrl_interval, :].reshape(-1)
            input_states = inputs.reshape((-1, self.optimize_dim))[
                :, -self.obs_dim :
            ].reshape(-1)
            return true_states - input_states

    def _trans_constraint_jac(
        self, inputs: np.ndarray, x: torch.Tensor, info: InfoDict
    ) -> np.ndarray:
        """
        Compute jacobian of transition constraint function (collocation only)
        """
        inputs = torch.tensor(inputs, dtype=torch.float32)
        jac = jacrev(self._trans_constraint_fcn)(inputs, x, info)
        return jac.numpy().astype("d")

    def _rollout(
        self, inputs: torch.Tensor, x: torch.Tensor, info: InfoDict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout furture states and rewards using environment model
        """
        st = time.time()
        self.system_simulations += 1
        inputs_repeated = inputs.reshape(
            (self.num_ctrl_points, self.optimize_dim)
        ).repeat_interleave(self.ctrl_interval, dim=0)
        states = torch.zeros((self.num_pred_step + 1, self.obs_dim))
        rewards = torch.zeros(self.num_pred_step)
        infos = {}
        states[0, :] = x
        done = torch.tensor([False])

        while True:
            if self.rollout_mode == "loop":
                next_x = x.unsqueeze(0)
                batched_info = {}
                if info:
                    batched_info = {key: value.unsqueeze(0) for key, value in info.items()}
                    infos = {key: [value.unsqueeze(0)] for key, value in info.items()}
                for i in np.arange(0, self.num_pred_step):
                    u = inputs_repeated[i, : self.action_dim].unsqueeze(0)
                    next_x, reward, done, batched_info = self.model.forward(
                        next_x, u, done=done, info=batched_info
                    )
                    rewards[i] = -reward * (self.gamma ** i)
                    states[i + 1, :] = next_x
                    if info:
                        for key in info.keys():
                            infos[key].append(batched_info[key])
                break

            elif self.rollout_mode == "batch":
                try:
                    xs = torch.cat(
                        (
                            x.unsqueeze(0),
                            inputs_repeated[
                                : -self.ctrl_interval : self.ctrl_interval,
                                -self.obs_dim :,
                            ],
                        )
                    )
                    us = inputs_repeated[:: self.ctrl_interval, : self.action_dim]
                    for i in range(self.ctrl_interval):
                        xs, rewards[i :: self.ctrl_interval], _, _ = self.model.forward(
                            xs, us, done=done, info={}
                        )
                        states[i + 1 :: self.ctrl_interval, :] = xs
                    rewards = -rewards * torch.logspace(
                        0, self.num_pred_step - 1, self.num_pred_step, base=self.gamma
                    )
                    break
                except (KeyError):
                    # model requires additional info to forward, can't use batch rollout mode
                    self.rollout_mode = "loop"
        et = time.time()
        self.system_simulation_time += et - st
        return states, rewards, infos

    def _compute_cost(
        self, inputs: torch.Tensor, x: torch.Tensor, info: InfoDict
    ) -> torch.Tensor:
        """
        Compute total cost of optimization inputs
        """
        # rollout states and rewards
        states, rewards, _ = self._rollout(inputs, x, info)

        # sum up integral costs from timestep 0 to T-1
        cost = torch.sum(rewards)

        # Terminal cost for timestep T
        if self.terminal_cost is not None:
            terminal_cost = self.terminal_cost(states[-1, :])
            cost += terminal_cost * (self.gamma ** self.num_pred_step)
        return cost

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

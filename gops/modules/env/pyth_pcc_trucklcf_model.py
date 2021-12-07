#   Copyright (c) 2020 ocp-tools Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Author: Fawang Zhang

# env.py
# Continuous version of PCC Truck

import math
import warnings
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class RoadMap():
    def __init__(self):
        self.path = os.path.join(os.path.dirname(__file__),"resources/G1511.csv")
        self.map_road = pd.DataFrame(pd.read_csv(self.path, header=None))
        self.x = np.array(self.map_road.iloc[0:, 0].dropna(), dtype='float32')  # x
        self.y = np.array(self.map_road.iloc[0:, 1], dtype='float32')  # theta
        self.z = np.array(self.map_road.iloc[0:, 2], dtype='float32')  # tan(theta)

    def load_data_cal(self):
        Theta = [0]
        for i in range(len(self.y)):
            if i > 0:
                delta = math.atan((self.z[i] - self.z[i - 1]) / (self.y[i] - self.y[i - 1]))
                Theta.append(delta)
        Theta_filter = savgol_filter(Theta, 35, 1, mode='nearest') #rad
        return self.y.tolist(), Theta_filter

class PythPccTrucklcfModel(torch.nn.Module):
    
    def __init__(self,road_map=RoadMap(),**kwargs):
        super().__init__()
        self.sample_batch_size = kwargs['sample_batch_size']
        self.dynamic_state_dim = kwargs['dynamic_state_dim']
        self.Np = kwargs['pre_horizon']
        self.v_target = 23 #[m/s]
        # dynamic model parameters
        self.r = 0.51 # wheel radius [m]
        self.m = 4455 + 570 + 735 # mass  [kg]
        self.rho = 1.206  # air density [kg/m**3]
        self.A = 6.8  # frontal area [m**2]
        self.Cd = 0.64  # drag coefficient
        self.f = 0.0015  # road  friction coefficient
        self.gear_set = {"1": 3.74, "2": 2.0, "3": 1.34, "4": 1.0, "5": 0.77, "6": 0.63}
        self.i0 = 5.571  # final gear ratio
        self.dynamic_T = 0.05
        self.step_T = 0.05
        self.gravity = 9.8
        self.etaT = 0.92 * 0.99  # efficiency
        self.tau_eng = 0.55  # time constant for engine dynamic
        self.EngSpd_Lower = 400  # engine speed lower limit [rpm]
        self.EngSpd_Upper = 3150  # engine speed upper limit [rpm]
        self.TransSpd_Lower = 400  # Transmission speed lower limit [rpm]
        self.TransSpd_Upper = 3150  # Transmission speed Upper limit [rpm]
        self.min_action = -63.6  # Tecmd lower limit [Nm]
        self.max_action = 603.9  # Tecmd Upper limit [Nm]
        self.ades_k = 0
        self.gear = torch.torch.full([self.sample_batch_size, ], 6, dtype=torch.int8)
        self.shift_interval = 4  # [s]
        # Road map
        self.map_x, self.map_Theta = road_map.load_data_cal()
        # speed at which to fail the episode
        self.v_threshold = 10 / 3.6  # error to v_target [m/s]
        slope_low = [-0.13]  # [rad]
        slope_high = [0.17]
        # is still within bounds
        lb_state = np.array([-self.v_threshold] + slope_low*self.Np)
        hb_state = np.array([self.v_threshold] + slope_high*self.Np)
        lb_action = np.array([self.min_action])
        hb_action = np.array([self.max_action])
        self.viewer = None
        self.steps_beyond_done = None
        self.Reset_iteration = 500
        self.iteration_index = 0

        # read torque map table
        self.path_Torque_trucksim = os.path.join(os.path.dirname(__file__), "resources/Te_throttle_engspd_150kw.csv")
        self.Torque_trucksim = pd.DataFrame(pd.read_csv(self.path_Torque_trucksim, header=None))
        self.Torque_throttle = np.array(self.Torque_trucksim.iloc[0, 1:].dropna())  # axis X: throttle
        self.Torque_engspd = np.array(self.Torque_trucksim.iloc[1:, 0])  # axis Y: engspd unit: rpm
        self.Torque = np.array(self.Torque_trucksim.iloc[1:, 1:])  # axis Z: Engine Torque, unit: N.m

        # do not change the following section
        self.register_buffer('lb_state', torch.tensor(lb_state, dtype=torch.float32))
        self.register_buffer('hb_state', torch.tensor(hb_state, dtype=torch.float32))
        self.register_buffer('lb_action', torch.tensor(lb_action, dtype=torch.float32))
        self.register_buffer('hb_action', torch.tensor(hb_action, dtype=torch.float32))
        self.initialize_state()
    
    def initialize_state(self):
        self.d_init_state = torch.empty([self.sample_batch_size, self.dynamic_state_dim])
        self.d_init_state[:, 0] = torch.normal(0.0, 0.35, [self.sample_batch_size, ]) +self.v_target  # v
        self.d_init_state[:, 1] = torch.linspace(self.min_action, self.max_action, self.sample_batch_size)
        self.d_init_state[:, 2] = torch.linspace(0, 10000,self.sample_batch_size)
        return self.d_init_state

    def forward(self, state: torch.Tensor, action: torch.Tensor, beyond_done=torch.tensor(0)):
        """
        rollout the model one step, notice this method will not change the value of self.state
        you need to define your own state transition  function here
        notice that all the variables contains the batch dim you need to remember this point
        when constructing your function
        :param state: datatype:torch.Tensor, shape:[sample_batch_size, state_dim]
        :param action: datatype:torch.Tensor, shape:[sample_batch_size, action_dim]
        :param beyond_done: flag indicate the state is already done which means it will not be calculated by the model
        :return:
                next_state:  datatype:torch.Tensor, shape:[sample_batch_size, state_dim]
                              the state will not change anymore when the corresponding flag done is set to True
                reward:  datatype:torch.Tensor, shape:[sample_batch_size, 1]
                isdone:   datatype:torch.Tensor, shape:[sample_batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        warning_msg = "action out of action space!"
        if not ((action <= self.hb_action).all() and (action >= self.lb_action).all()):
            warnings.warn(warning_msg)
            action = clip_by_tensor(action, self.lb_action, self.hb_action)

        warning_msg = "state out of state space!"
        if not ((state <= self.hb_state).all() and (state >= self.lb_state).all()):
            warnings.warn(warning_msg)
            state = clip_by_tensor(state, self.lb_state, self.hb_state)

        d_v, d_Te, d_x = self.dynamic_state[:, 0], self.dynamic_state[:, 1], self.dynamic_state[:, 2]

        # update dynamic state
        theta = torch.from_numpy(np.interp(d_x.detach().numpy(), self.map_x, self.map_Theta))
        # 换挡等
        if self.iteration_index % self.shift_interval == 0:
            self.ig = self.Gear_Ratio(self.gear)
            self.EngSpd, self.TransSpd = self.Cal_EngSpd(d_v, self.ig)
            self.pedal_percent = self.Thr_calc(self.EngSpd, action)
            up_th, down_th = self.Threshold_LU(self.gear, self.pedal_percent)
            self.gear = self.Shift_Logic(self.gear, self.TransSpd, up_th, down_th)
            self.iteration_index = 0
        else:
            self.ig = self.Gear_Ratio(self.gear)

        d_v_next = d_v + self.dynamic_T * (
                d_Te * self.ig * self.i0 * self.etaT / (self.r * self.m) - 0.5 * self.Cd * self.A * self.rho * (
                d_v ** 2) / self.m - self.gravity * (
                        self.f * torch.cos(theta) + torch.sin(theta)))
        d_Te_next = d_Te + (action[:, 0] - d_Te) / self.tau_eng * self.dynamic_T
        d_x_next = d_x + self.dynamic_T * d_v
        dynamic_state_next = torch.stack([d_v_next, d_Te_next, d_x_next]).transpose(1, 0)
        self.dynamic_state = self.check_done(dynamic_state_next)

        # compute future road slope
        future_ds_list = [d_x_next + d_v_next * self.dynamic_T * i for i in range(self.Np)]
        future_thetas_list = np.array(
            [np.interp(future_ds_list[i].detach().numpy(), self.map_x, self.map_Theta) for i in range(self.Np)],
            dtype=float).T
        # future_thetas_list = np.zeros((self.Np, self.sample_batch_size))
        # for i in range(self.Np):
        #     future_thetas_list[i] = np.interp(future_ds_list[i].detach().numpy(), self.map_x, self.map_Theta)
        # future_thetas_list = future_thetas_list.T

        #network input
        state_next = torch.empty([self.sample_batch_size, 1 + self.Np])
        state_next[:, 0] = d_v_next - self.v_target # v_error [-3~3]
        state_next[:, 1:self.Np+1] = torch.tensor(future_thetas_list)
        ############################################################################################
        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        # weighting
        weight_fuel = 4e-2
        weight_tracking = 1
        weight_control = 2.5e-5
        # fuel model fitting coefficient
        p00 = 2.441
        p10 = -0.001083
        p01 = -0.003831
        p20 = -1.858e-05
        p11 = 1.13e-05
        p02 = 1.408e-06
        x = d_Te
        y = 60 * self.i0 * self.ig * d_v / (2 * math.pi * self.r)
        self.Qfuel = p00 + p10 * x + p01 * y + p20 * (x ** 2) + p11 * x * y + p02 * (y ** 2)
        # reward
        reward = -(weight_fuel * self.Qfuel + weight_tracking * (
                (d_v-self.v_target) ** 2) + weight_control * ((action - self.ades_k) ** 2))
        self.ades_k = action.detach()

        ############################################################################################
        # define the ending condation here the format is just like isdone = l(next_state)
        isdone = beyond_done #state[:, 0].new_zeros(size=[state.size()[0]], dtype=torch.bool)  todo
        # isdone = state_next[:, 0] < -self.v_threshold \
        #        or state_next[:, 0] > self.v_threshold \
        #        or d_x_next < 0 \
        #        or d_x_next > 10000
        # isdone = bool(isdone)
        self.iteration_index += 1
        return state_next, reward, isdone

    def forward_n_step(self,state: torch.Tensor, action, step, beyond_done=torch.tensor(0)):
        self.dynamic_state = self.d_init_state
        d_v, d_Te, d_x = self.dynamic_state[:, 0], self.dynamic_state[:, 1], self.dynamic_state[:, 2]

        # compute future road slope
        future_ds_list = [d_x + d_v * self.dynamic_T * i for i in range(self.Np)]
        future_thetas_list = np.array([np.interp(future_ds_list[i].detach().numpy(), self.map_x, self.map_Theta) for i in range(self.Np)],dtype=float).T
        # future_thetas_list = np.zeros((self.Np, self.sample_batch_size))
        # for i in range(self.Np):
        #     future_thetas_list[i] = np.interp(future_ds_list[i].detach().numpy(), self.map_x, self.map_Theta)
        # future_thetas_list = future_thetas_list.T
        # network input
        state = torch.empty([self.sample_batch_size, 1 + self.Np])
        state[:, 0] = d_v - self.v_target  # v_error [-3~3]
        state[:, 1:self.Np + 1] = torch.tensor(future_thetas_list)
        next_state_list = []
        done_list = []
        v_pi = 0.0
        for i in range(step):
            state, reward, beyond_done = self.forward(state, action(state), beyond_done)
            v_pi = v_pi + reward
            next_state_list.append(state)
            done_list.append(beyond_done)
        return next_state_list, v_pi, done_list

    def _reset_state(self,state: torch.Tensor):
        """
        reset state to initial state.
        Parameters
        ----------
        state: tensor   shape: [sample_batch_size, STATE_DIMENSION]
            state used for checking.
        Returns
        -------
        state: state after reset.
        """
        for i in range(self.sample_batch_size):
            if self._reset_index[i] == 1:
                state[i, :] = self.d_init_state[i, :]
        return state

    def check_done(self, state):
        """
        Check if the states reach unreasonable zone and reset them
        Parameters
        ----------
        state: tensor   shape: [sample_batch_size, STATE_DIMENSION]
            state used for checking.
        Returns
        -------
        """
        # [256 * 1] Kron [1 * 1] <np, 256 * 1>
        threshold = np.kron(np.ones([self.sample_batch_size, 1]),
                            np.array([self.v_threshold]))  # 矩阵乘法 A_n×m*B_p×q=Cnp*mq ,self.x_range
        threshold = np.array(threshold, dtype='float32')
        check_state = state[:, 0].clone() - self.v_target
        check_state.detach_()
        sign_error = torch.sign(torch.abs(check_state) - threshold)  # if abs state is over threshold, sign_error = 1
        self._reset_index, _ = torch.max(sign_error, 1)  # if one state is over threshold, _reset_index = 1
        for i in range(self.sample_batch_size):
            if state[i, 2] < 0 or state[i, 2] > 10000:
                self._reset_index[i] = 1
        # if self.iteration_index == self.Reset_iteration:
        #     self._reset_index = torch.from_numpy(np.ones([self.sample_batch_size,],dtype='float32'))
        #     self.iteration_index = 0
        reset_state = self._reset_state(state)
        return reset_state

    def Cal_EngSpd(self, v, ig):
        engSpd = v * ig * self.i0 * 60 / (2*self.r * math.pi)
        self.EngSpd = clip_by_tensor(engSpd,self.EngSpd_Lower,self.EngSpd_Upper)
        TransSpd = v * 60/(2 * self.r * math.pi) * self.i0
        return self.EngSpd, TransSpd

    def Gear_Ratio(self, gear):
        gear_ratio = torch.zeros(self.sample_batch_size)
        for i in range(len(gear)):
            gear_ratio = self.gear_set[str(int(gear[i].item()))]
        return gear_ratio

    def Thr_calc(self, EngSpd, EngT):
        throttle = 0.0  # default
        torq_temp = [0] * 11
        EngSpd = clip_by_tensor(EngSpd, self.EngSpd_Lower, self.EngSpd_Upper)
        EngT = clip_by_tensor(EngT, self.min_action, self.max_action)
        Pedal_Percent = torch.zeros(self.sample_batch_size)
        for i in range(len(EngT)):
            We = EngSpd[i].item()
            Te = EngT[i].item()
            index_rpm = np.argwhere(We >= self.Torque_engspd)[-1][0]
            if We == self.Torque_engspd[-1]:
                index_rpm = 16
                scale_rpm = 0
            else:
                scale_rpm = (We - self.Torque_engspd[index_rpm]) / (
                            self.Torque_engspd[index_rpm + 1] - self.Torque_engspd[index_rpm])

            for mm in range(len(self.Torque_throttle)):
                if We == self.Torque_engspd[-1]:
                    torq_temp[mm] = self.Torque[index_rpm][mm]
                else:
                    torq_temp[mm] = self.Torque[index_rpm][mm] + scale_rpm * (
                                self.Torque[index_rpm + 1][mm] - self.Torque[index_rpm][mm])

            for mm in range(10):
                if Te >= torq_temp[mm] and Te < torq_temp[mm + 1]:
                    index_torq = mm
                    break
            if Te < torq_temp[0]:
                index_torq = 0
            if Te >= torq_temp[10]:
                index_torq = 10

            if index_torq == 0:
                if torq_temp[1] - torq_temp[0] == 0.0:
                    throttle = 0.0 + (Te - torq_temp[0]) / (0.0000001) * (0.1 - 0)
                else:
                    throttle = 0.0 + (Te - torq_temp[0]) / (torq_temp[1] - torq_temp[0]) * (0.1 - 0)
            elif index_torq == 1:
                throttle = 0.1 + (Te - torq_temp[1]) / (torq_temp[2] - torq_temp[1]) * (0.2 - 0.1)
            elif index_torq == 2:
                throttle = 0.2 + (Te - torq_temp[2]) / (torq_temp[3] - torq_temp[2]) * (0.3 - 0.2)
            elif index_torq == 3:
                throttle = 0.3 + (Te - torq_temp[3]) / (torq_temp[4] - torq_temp[3]) * (0.4 - 0.3)
            elif index_torq == 4:
                throttle = 0.4 + (Te - torq_temp[4]) / (torq_temp[5] - torq_temp[4]) * (0.5 - 0.4)
            elif index_torq == 5:
                throttle = 0.5 + (Te - torq_temp[5]) / (torq_temp[6] - torq_temp[5]) * (0.6 - 0.5)
            elif index_torq == 6:
                throttle = 0.6 + (Te - torq_temp[6]) / (torq_temp[7] - torq_temp[6]) * (0.7 - 0.6)
            elif index_torq == 7:
                throttle = 0.7 + (Te - torq_temp[7]) / (torq_temp[8] - torq_temp[7]) * (0.8 - 0.7)
            elif index_torq == 8:
                throttle = 0.8 + (Te - torq_temp[8]) / (torq_temp[9] - torq_temp[8]) * (0.9 - 0.8)
            elif index_torq == 9:
                throttle = 0.9 + (Te - torq_temp[8]) / (torq_temp[9] - torq_temp[8]) * (1.0 - 0.9)
            elif index_torq == 10:
                throttle = 1.0
            if throttle > 1.0:
                throttle = 1.0
            if throttle < 0.0:
                throttle = 0.0
            Pedal_Percent[i] = throttle
        return Pedal_Percent

    def Shift_Logic(self, gear_input, AT_Speed, up_th, down_th):
        gear_output = gear_input.clone()
        for i in range(len(gear_input)):
            geari = int(gear_input[i].item())
            if geari == 1:
                if AT_Speed[i].item() < down_th[i].item():
                    self.gear = 1
                elif down_th[i].item() >= AT_Speed[i].item() and AT_Speed[i].item() < up_th[i].item():
                    self.gear = 1
                elif AT_Speed[i].item() >= up_th[i].item():
                    self.gear = 2

            elif geari == 2:
                if AT_Speed[i].item() < down_th[i].item():
                    self.gear = 1
                elif down_th[i].item() >= AT_Speed[i].item() and AT_Speed[i].item() < up_th[i].item():
                    self.gear = 2
                elif AT_Speed[i].item() >= up_th[i].item():
                    self.gear = 3

            elif geari == 3:
                if AT_Speed[i].item() < down_th[i].item():
                    self.gear = 2
                elif down_th[i].item() >= AT_Speed[i].item() and AT_Speed[i].item() < up_th[i].item():
                    self.gear = 3
                    # return gear
                elif AT_Speed[i].item() >= up_th[i].item():
                    self.gear = 4

            elif geari == 4:
                if AT_Speed[i].item() < down_th[i].item():
                    self.gear = 3
                elif down_th[i].item() >= AT_Speed[i].item() and AT_Speed[i].item() < up_th[i].item():
                    self.gear = 4
                elif AT_Speed[i].item() >= up_th[i].item():
                    self.gear = 5

            elif geari == 5:
                if AT_Speed[i].item() < down_th[i].item():
                    self.gear = 4
                elif down_th[i].item() >= AT_Speed[i].item() and AT_Speed[i].item() < up_th[i].item():
                    self.gear = 5
                elif AT_Speed[i].item() >= up_th[i].item():
                    self.gear = 6
            else:
                if AT_Speed[i].item() < down_th[i].item():
                    self.gear = 5
                else:
                    self.gear = 6
            gear_output[i] = self.gear
        return gear_output

    def Threshold_LU(self, gear_input: torch.Tensor, throttle: torch.Tensor):
        up_th = torch.zeros(self.sample_batch_size)
        down_th = torch.zeros(self.sample_batch_size)
        for i in range(len(gear_input)):
            geari = int(gear_input[i].item())
            if geari == 1:  # 1-2
                up = np.interp(throttle[i].item(), [0, 0.2, 0.8, 1], [158, 158, 675, 675])
                down = np.interp(throttle[i].item(), [0, 0.2, 0.8, 1], [70, 70, 110, 110])
            elif geari == 2:
                up = np.interp(throttle[i].item(), [0, 0.2, 0.8, 1], [237, 237, 1255, 1255])
                down = np.interp(throttle[i].item(), [0, 0.2, 0.8, 1], [213, 213, 450, 450])
            elif geari == 3:
                up = np.interp(throttle[i].item(), [0, 0.2, 0.8, 1], [356, 356, 1860, 1860])
                down = np.interp(throttle[i].item(), [0, 0.2, 0.8, 1], [320, 320, 710, 710])
            elif geari == 4:
                up = np.interp(throttle[i].item(), [0, 0.2, 0.8, 1], [633, 633, 2490, 2490])
                down = np.interp(throttle[i].item(), [0, 0.75, 0.9, 1], [480, 480, 1067, 1067])
            elif geari == 5:
                up = np.interp(throttle[i].item(), [0, 0.2, 0.8, 1], [800, 800, 3080, 3080])
                down = np.interp(throttle[i].item(), [0, 0.75, 0.9, 1], [620, 620, 1250, 1250])
            else:
                up = np.interp(throttle[i].item(), [0, 0.2, 0.8, 1], [1067, 1067, 3080, 3080])
                down = np.interp(throttle[i].item(), [0, 0.75, 0.9, 1], [960, 960, 1250, 1250])
            up_th[i] = up
            down_th[i] = down
        return up_th, down_th

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min) * t + (t < t_min) * t_min # if t>= t_min:t else t_min
    result = (result <= t_max) * result + (result > t_max) * t_max
    return result


if __name__ == "__main__":
    print(111111111)
    # parser = argparse.ArgumentParser()
    # ################################################
    # # 1. Parameters for environment
    # parser.add_argument('--dynamic_state_dim', type=int, default=3)
    # parser.add_argument('--pre_horizon', type=int, default=10)
    # parser.add_argument('--sample_batch_size', type=int, default=256)
    # args = vars(parser.parse_args())
    # env = PythPccTrucklcfModel(**args)
    # env.initialize_state()
    # for i in range(10000):
    #     next_state, reward, done = env.forward(state=torch.zeros(256,11), action=torch.linspace(-63.6, 603.9, 256), beyond_done=torch.tensor(1))
    #     print(next_state)
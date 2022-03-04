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
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from scipy.signal import savgol_filter
import os
import pandas as pd
from gym.wrappers.time_limit import TimeLimit

class RoadMap():
    def __init__(self):
        self.path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"resources/pyth_pcc_trucklcf_file/Roadmap.csv")
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

class PythPCCTruck(gym.Env):
    def __init__(self,road_map=RoadMap(),**kwargs):
        self.v_target = 23 #[m/s]
        self.Np = kwargs['pre_horizon']
        self.max_episode_steps = kwargs['max_iteration']
        # init state
        self.dynamic_state = [23, 0, 0]
        # dynamic model parameters
        self.r = 0.51 # wheel radius [m]
        self.m = 4455 + 570 + 735 # mass  [kg]
        self.rho = 1.206  # air density [kg/m**3]
        self.A = 6.8  # frontal area [m**2]
        self.Cd = 0.64  # drag coefficient
        self.f = 0.0015  # road  friction coefficient
        self.gear_set = {"1": 3.74, "2": 2.0, "3": 1.34, "4": 1.0, "5": 0.77, "6": 0.63} # gear ratio set
        self.i0 = 5.571  # final gear ratio
        self.dynamic_T = 0.05
        self.step_T = 0.05
        self.gravity = 9.81
        self.etaT = 0.92 * 0.99  # efficiency
        self.tau_eng = 0.55 # time constant for engine dynamic
        self.EngSpd_Lower = 400  # engine speed lower limit [rpm]
        self.EngSpd_Upper = 3150 # engine speed upper limit [rpm]
        self.TransSpd_Lower = 400  # Transmission speed lower limit [rpm]
        self.TransSpd_Upper = 3150  # Transmission speed Upper limit [rpm]
        self.min_action = -63.6  # Tecom lower limit [Nm]
        self.max_action = 603.9  # Tecom Upper limit [Nm]
        self.ades_k = 0
        self.gear = 6
        # Road Slope
        self.map_x, self.map_Theta = road_map.load_data_cal()

        # speed at which to fail the episode
        self.v_threshold = 10 / 3.6 # error to v_target [m/s]
        slope_low = [-0.13]  # [rad]
        slope_high = [0.17]
        # is still within bounds
        state_low = np.array([-self.v_threshold] + slope_low*int(self.Np))
        state_high = np.array([self.v_threshold] + slope_high*int(self.Np))
        action_low = np.array([self.min_action])
        action_high = np.array([self.max_action])
        self.observation_space = spaces.Box(state_low, state_high)
        self.action_space = spaces.Box(action_low, action_high)
        self.seed()
        self.viewer = None
        self.steps_beyond_done = None
        self.steps = 0
        # read fuel map table and torque map table
        self.path_Fuel_trucksim = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/pyth_pcc_trucklcf_file/Fuel_we_thr_150kw.csv")
        self.Fuel_trucksim = pd.DataFrame(pd.read_csv(self.path_Fuel_trucksim, header=None))
        self.Fuel_throttle = np.array(self.Fuel_trucksim.iloc[0, 1:].dropna())  # axis X: throttle
        self.Fuel_engspd = np.array(self.Fuel_trucksim.iloc[1:, 0])  # axis Y: engspd unit: rpm
        self.Fuel = np.array(self.Fuel_trucksim.iloc[1:, 1:])  # axis Z: Fuel Rate, unit: kg/sec
        self.path_Torque_trucksim = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/pyth_pcc_trucklcf_file/Te_throttle_engspd_150kw.csv")
        self.Torque_trucksim = pd.DataFrame(pd.read_csv(self.path_Torque_trucksim, header=None))
        self.Torque_throttle = np.array(self.Torque_trucksim.iloc[0, 1:].dropna())  # axis X: throttle
        self.Torque_engspd = np.array(self.Torque_trucksim.iloc[1:, 0])  # axis Y: engspd unit: rpm
        self.Torque = np.array(self.Torque_trucksim.iloc[1:, 1:])  # axis Z: Engine Torque, unit: N.m

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, action):
        v, Te, x = self.dynamic_state
        theta = np.interp(x, self.map_x, self.map_Theta)
        #换挡等
        self.ig = self.Gear_Ratio(self.gear)
        self.EngSpd, self.TransSpd = self.Cal_EngSpd(v, self.ig)
        self.pedal_percent = self.Thr_calc(self.EngSpd, action)
        up_th, down_th = self.Threshold_LU(self.gear, self.pedal_percent)
        self.gear = self.Shift_Logic(self.gear, self.TransSpd, up_th, down_th)

        v_next = v + self.dynamic_T * (
                    Te * self.ig * self.i0 * self.etaT / (self.r * self.m) - 0.5 * self.Cd * self.A * self.rho * (v**2) / self.m - self.gravity * (
                        self.f * math.cos(theta) + math.sin(theta)))
        Te_next = Te + (action - Te) / self.tau_eng * self.dynamic_T
        x_next = x + self.dynamic_T * v
        self.dynamic_state = v_next, Te_next, x_next

        return (v_next, float(Te_next), x_next)

    def step(self, action=None):
        v, Te, x = self.stepPhysics(action)
        self.ades_k = action
        done = v - self.v_target < - self.v_threshold \
               or v - self.v_target > self.v_threshold \
               or x < 0 \
               or x > 10000
        done = bool(done)

        # compute future road slope
        future_ds_list = [x + v * self.dynamic_T * i for i in range(self.Np)]
        future_thetas_list = np.interp(future_ds_list, self.map_x, self.map_Theta)
        self.state = np.concatenate((np.array([v - self.v_target]), future_thetas_list), axis=0).flatten(order='C')

        # -----------------
        self.steps += 1
        if self.steps >= self.max_episode_steps:
            done = True

        # reward
        weight_fuel = 4e-2
        weight_tracking = 1
        weight_control = 2.5e-5
        self.Qfuel = self.Engine_Fuel_Model(self.EngSpd,float(self.pedal_percent))
        if not done:
            reward = -(weight_fuel * self.Qfuel + weight_tracking * (
                        (v - self.v_target) ** 2) + weight_control * ((action - self.ades_k) ** 2))
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = -(weight_fuel * self.Qfuel + weight_tracking * (
                        (v - self.v_target) ** 2) + weight_control * ((action - self.ades_k) ** 2))
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn("""
                You are calling 'step()' even though this environment has already returned
                done = True. You should always call 'reset()' once you receive 'done = True'
                Any further steps are undefined behavior.
                    """)
            self.steps_beyond_done += 1
            reward = -10000.0
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.zeros(int(self.Np)+1, dtype='float')
        self.steps_beyond_done = None
        self.steps = 0
        return np.array(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()

    def Cal_EngSpd(self, v, ig):
        EngSpd_Lower = self.EngSpd_Lower
        EngSpd_Upper = self.EngSpd_Upper
        self.EngSpd = float(v) * ig * self.i0 * 60 / (2*self.r * math.pi)
        if self.EngSpd > EngSpd_Upper:
            self.EngSpd = EngSpd_Upper
        if self.EngSpd < EngSpd_Lower:
            self.EngSpd = EngSpd_Lower
        TransSpd = float(v) * 60/(2 * self.r * math.pi) * self.i0
        return self.EngSpd, TransSpd

    def Engine_Fuel_Model(self, EngSpd, throttle):
        # 用节气门位置和转速查转距表
        # trucksim fuel model
        if EngSpd < self.Fuel_engspd[0] or EngSpd > self.Fuel_engspd[-1]:
            if EngSpd < self.Fuel_engspd[0]:
                EngSpd = self.Fuel_engspd[0]
            if EngSpd > self.Fuel_engspd[-1]:
                EngSpd = self.Fuel_engspd[-1]
        if throttle < self.Fuel_throttle[0] or throttle > self.Fuel_throttle[-1]:
            if throttle < self.Fuel_throttle[0]:
                throttle = self.Fuel_throttle[0]
            if throttle > self.Fuel_throttle[-1]:
                throttle = self.Fuel_throttle[-1]
        Thr_index = int(np.argwhere(throttle >= self.Fuel_throttle)[-1])
        EngSpd_index = int(np.argwhere(EngSpd >= self.Fuel_engspd)[-1])
        if Thr_index != int(len(self.Fuel_throttle) - 1) and EngSpd_index != int(len(self.Fuel_engspd) - 1):
            scale_thr = (throttle - self.Fuel_throttle[Thr_index]) / (
                        self.Fuel_throttle[Thr_index + 1] - self.Fuel_throttle[Thr_index])
            scale_rpm = (EngSpd - self.Fuel_engspd[EngSpd_index]) / (
                        self.Fuel_engspd[EngSpd_index + 1] - self.Fuel_engspd[EngSpd_index])
            Fuel_temp1 = self.Fuel[EngSpd_index][Thr_index] + scale_thr * (
                    self.Fuel[EngSpd_index][Thr_index + 1] - self.Fuel[EngSpd_index][Thr_index])
            Fuel_temp2 = self.Fuel[EngSpd_index + 1][Thr_index] + scale_thr * (
                    self.Fuel[EngSpd_index + 1][Thr_index + 1] - self.Fuel[EngSpd_index + 1][Thr_index])
            Fuel_result = Fuel_temp1 + scale_rpm * (Fuel_temp2 - Fuel_temp1)

        elif Thr_index == int(len(self.Fuel_throttle) - 1) and EngSpd_index != int(len(self.Fuel_engspd) - 1):
            scale_rpm = (EngSpd - self.Fuel_engspd[EngSpd_index]) / (
                        self.Fuel_engspd[EngSpd_index + 1] - self.Fuel_engspd[EngSpd_index])
            Fuel_result = self.Fuel[EngSpd_index][Thr_index] + scale_rpm * (
                    self.Fuel[EngSpd_index + 1][Thr_index] - self.Fuel[EngSpd_index][Thr_index])

        elif Thr_index != int(len(self.Fuel_throttle) - 1) and EngSpd_index == int(len(self.Fuel_engspd) - 1):
            scale_thr = (throttle - self.Fuel_throttle[Thr_index]) / (
                        self.Fuel_throttle[Thr_index + 1] - self.Fuel_throttle[Thr_index])
            Fuel_result = self.Fuel[EngSpd_index][Thr_index] + scale_thr * (
                    self.Fuel[EngSpd_index][Thr_index + 1] - self.Fuel[EngSpd_index][Thr_index])

        else:
            Fuel_result = self.Fuel[EngSpd_index][Thr_index]
        return Fuel_result * 1000

    def Gear_Ratio(self, gear):
        gear_ratio = self.gear_set[str(gear)]
        return gear_ratio

    def Thr_calc(self, We, Te):
        EngSpd_Lower = self.EngSpd_Lower
        EngSpd_Upper = self.EngSpd_Upper
        Tecom_Lower = self.min_action
        Tecom_Upper = self.max_action
        throttle = 0.0  # default
        torq_temp = [0] * 11

        if We < EngSpd_Lower or We > EngSpd_Upper:
            # print("ENGINE SPEED INPUT WRONG(thr_cal)!!", We)
            if We < EngSpd_Lower:
                We = EngSpd_Lower
            else:
                We = EngSpd_Upper
        if Te < Tecom_Lower or Te > Tecom_Upper:
            # print("TORQUE INPUT WRONG!!", Te)
            if Te < Tecom_Lower:
                Te = Tecom_Lower
            else:
                Te = Tecom_Upper

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
        return throttle

    def Shift_Logic(self, gear,AT_Speed, up_th, down_th):  # 根据变速箱转速判断档位
        if gear == 1:
            if AT_Speed < down_th:
                gear = 1
            elif AT_Speed >= down_th and AT_Speed < up_th:
                gear = 1
            elif AT_Speed >= up_th:
                gear = 2
        elif gear == 2:
            if AT_Speed < down_th:
                gear = 1
            elif AT_Speed >= down_th and AT_Speed < up_th:
                gear = 2
            elif AT_Speed >= up_th:
                gear = 3
        elif gear == 3:
            if AT_Speed < down_th:
                gear = 2
            elif AT_Speed >= down_th and AT_Speed < up_th:
                gear = 3
            elif AT_Speed >= up_th:
                gear = 4

        elif gear == 4:
            if AT_Speed < down_th:
                gear = 3
            elif AT_Speed >= down_th and AT_Speed < up_th:
                gear = 4
            elif AT_Speed >= up_th:
                gear = 5

        elif gear == 5:
            if AT_Speed < down_th:
                gear = 4
            elif AT_Speed >= down_th and AT_Speed < up_th:
                gear = 5
            elif AT_Speed >= up_th:
                gear = 6
        else:
            if AT_Speed < down_th:
                gear = 5
            else:
                gear = 6
        return gear

    def Threshold_LU(self,gear,throttle): #
        if gear == 1:
            up_th = np.interp(throttle, [0, 0.2, 0.8, 1], [158, 158, 675, 675])
            down_th = np.interp(throttle, [0, 0.2, 0.8, 1], [70, 70, 110, 110])
        elif gear == 2:
            up_th = np.interp(throttle, [0, 0.2, 0.8, 1], [237, 237, 1255, 1255])
            down_th = np.interp(throttle, [0, 0.2, 0.8, 1], [213, 213, 450, 450])
        elif gear == 3:
            up_th = np.interp(throttle, [0, 0.2, 0.8, 1], [356, 356, 1860, 1860])
            down_th = np.interp(throttle, [0, 0.2, 0.8, 1], [320, 320, 710, 710])
        elif gear == 4:
            up_th = np.interp(throttle, [0, 0.2, 0.8, 1], [633, 633, 2490, 2490])
            down_th = np.interp(throttle, [0, 0.75, 0.9, 1], [480, 480, 1067, 1067])
        elif gear == 5:
            up_th = np.interp(throttle, [0, 0.2, 0.8, 1], [800, 800, 3080, 3080])
            down_th = np.interp(throttle, [0, 0.75, 0.9, 1], [620, 620, 1250, 1250])
        else:
            up_th = np.interp(throttle, [0, 0.2, 0.8, 1], [800, 800, 3080, 3080])
            down_th = np.interp(throttle, [0, 0.75, 0.9, 1], [620, 620, 1250, 1250])
        return up_th, down_th

def env_creator(**kwargs):
    return TimeLimit(PythPCCTruck(**kwargs), 20000)

if __name__ == "__main__":
    env = PythPCCTruck(pre_horizon=10, max_iteration=10)
    env.reset()
    for i in range(10000):
        next_state, reward, done, _ = env.step(100)
        print(next_state)
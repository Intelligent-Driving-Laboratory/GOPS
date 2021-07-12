#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University

#  Creator: Fawang Zhang
#  Description: PCC Car CVT Model Environment

#  Update Date: 2021-07-011, Fawang Zhang: Solve the problem of slow training speed

import os
import gym
from gym import spaces
from gym.wrappers.time_limit import TimeLimit
import numpy as np
from ctypes import *
import math
import pandas as pd
from scipy.signal import savgol_filter


class CarParameter(Structure):
    """
    Car Position Structure for C/C++ interface
    """
    _fields_ = [
        ("LX_AXLE", c_float),  # 轴距，m
        ("LX_CG_SU", c_float),  # 悬上质量质心至前轴距离，m
        ("M_SU", c_float),  # 悬上质量，kg
        ("IZZ_SU", c_float),  # 转动惯量，kg*m^2
        ("A", c_float),  # 迎风面积，m^2
        ("CFx", c_float),  # 空气动力学侧偏角为零度时的纵向空气阻力系数
        ("AV_ENGINE_IDLE", c_float),  # 怠速转速，rpm
        ("IENG", c_float),  # 曲轴转动惯量，kg*m^2
        ("TAU", c_float),  # 发动机-变速箱输入轴 时间常数，s
        ("R_GEAR_TR1", c_float),  # 最低档变速箱传动比
        ("R_GEAR_FD", c_float),  # 主减速器传动比
        ("BRAK_COEF", c_float),  # 液压缸变矩系数,Nm/(MPa)
        ("Steer_FACTOR", c_float),  # 转向传动比
        ("M_US", c_float),  # 簧下质量，kg
        ("RRE", c_float),  # 车轮有效滚动半径，m
        ("CF", c_float),  # 前轮侧偏刚度，N/rad
        ("CR", c_float),  # 后轮侧偏刚度，N/rad
        ("ROLL_RESISTANCE", c_float)]  # 滚动阻力系数

class RoadParameter(Structure):
    """
    Car Position Structure for C/C++ interface
    """
    _fields_ = [("slope", c_float)]

class VehicleInfo(Structure):
    """车辆动力学参数结构体"""
    _fields_ = [
        ("AV_Eng", c_float),
        ("AV_Y", c_float),
        ("Ax", c_float),
        ("Ay", c_float),
        ("A", c_float),
        ("Beta", c_float),
        ("Bk_Pressure", c_float),
        ("Mfuel", c_float),  # 累计，g
        ("M_EngOut", c_float),
        ("Rgear_Tr", c_float),
        ("Steer_SW", c_float),
        ("StrAV_SW", c_float),
        ("Steer_L1", c_float),
        ("Throttle", c_float),
        ("Vx", c_float),
        ("Vy", c_float),
        ("Yaw", c_float),  # 偏航角, rad
        ("Qfuel", c_float),  # rate,g/s
        ("Mileage", c_float)]  # 里程, m

class RoadMap():
    def __init__(self):
        self.path = os.path.join(os.path.dirname(__file__),"resources/roadmap.csv")
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

class PythPCCCarCVTModel(gym.Env):

    def __init__(self, road_map=RoadMap(),Car_parameter=CarParameter(),**kwargs):
        self.x_map, self.Theta = road_map.load_data_cal()
        self.dll = CDLL(os.path.join(os.path.dirname(__file__),"resources/CarModel_CVT.dll"))
        #Car_parameter
        Car_parameter.LX_AXLE = 3.16 # 轴距，m
        Car_parameter.LX_CG_SU = 1.265 # 悬上质量质心至前轴距离，m
        Car_parameter.M_SU = 1820.0 # 悬上质量，kg
        Car_parameter.IZZ_SU = 4095.0 # 转动惯量，kg*m^2
        Car_parameter.A = 3.0 # 迎风面积，m^2
        Car_parameter.CFx = 0.3 # 空气动力学侧偏角为零度时的纵向空气阻力系数
        Car_parameter.AV_ENGINE_IDLE = 750 # 怠速转速，rpm
        Car_parameter.IENG = 0.4 # 曲轴转动惯量，kg*m^2
        Car_parameter.TAU = 0.3 # 发动机-变速箱输入轴 时间常数，s
        Car_parameter.R_GEAR_TR1 = 4.6 # 最低档变速箱传动比
        Car_parameter.R_GEAR_FD = 2.65 # 主减速器传动比
        Car_parameter.BRAK_COEF = 1100.0 # 液压缸变矩系数,Nm/(MPa)
        Car_parameter.Steer_FACTOR = 16.5 # 转向传动比
        Car_parameter.M_US = 200 # 簧下质量，kg
        Car_parameter.RRE = 0.353 # 车轮有效滚动半径，m
        Car_parameter.CF = -128915.5 # 前轮侧偏刚度，N/rad
        Car_parameter.CR = -117481.8 # 后轮侧偏刚度，N/rad
        Car_parameter.ROLL_RESISTANCE = 0.0041 # 滚动阻力系数

        #init state
        x = 0.
        self.ego_x = x
        y = 0.
        heading = 0.
        v = 20.
        self.step_length = 0.05
        self.dll.init(c_float(x), c_float(y),
                      c_float(heading), c_float(v),
                      c_float(self.step_length), byref(Car_parameter))
        self.v_target = 20 #target speed
        self.Np = 50 # prediciton horizon

        # state limit
        slope_low = [-0.06]
        slope_high = [0.06]
        state_low = np.array([-10.0]+slope_low*self.Np)
        state_high = np.array([10.0]+slope_high*self.Np)
        #action limit
        action_high = np.array([4.0])
        action_low = np.array([-6.0])
        self.action_space = spaces.Box(action_low,action_high)
        self.observation_space = spaces.Box(state_low, state_high)

        self.steps_beyond_done = None
        # self.max_episode_steps = 20000
        self.steps = 0
        self.car_info = VehicleInfo()  # 自车信息结构体
        self.ades_k = 0

    def step(self, action=None):
        ades = action
        steer_front = 0
        if ades is None:
            AD = c_float(self.ades)
        else:
            AD = c_float(ades)
            self.ades = AD
        if steer_front is None:
            SW = c_float(self.steer_front)
        else:
            SW = c_float(steer_front)
            self.steer_front = SW
        # input the data type of the state
        x = c_float(0.0)
        y = c_float(0.0)
        heading = c_float(0.0)  # rad
        acc = c_float(0.0)
        v = c_float(0.0)
        r = c_float(0.0)  # engine speed
        i = c_float(0.0)  # gear ratio

        # get and input road slope
        road_info = RoadParameter()
        road_info.slope = np.interp(self.ego_x, self.x_map, self.Theta)
        self.dll.sim(byref(road_info), byref(AD), byref(SW),
                     byref(x), byref(y), byref(heading), byref(acc), byref(v),
                     byref(r), byref(i))
        heading.value = (heading.value / math.pi * 180.0)

        (self.ego_x, self.ego_y, self.ego_vel, self.ego_heading, self.acc,
         self.engine_speed, self.drive_ratio) = (x.value, y.value, v.value,
                                                 heading.value, acc.value,
                                                 r.value, i.value)
        self.dll.get_info(byref(self.car_info))
        # compute future road slope
        future_ds_list = [self.ego_x + self.car_info.Vx * self.step_length * i for i in range(self.Np)]
        future_thetas_list = list(np.interp(future_ds_list,self.x_map,self.Theta))
        # next state
        self.state = [self.car_info.Vx-self.v_target] + future_thetas_list

        done =self.car_info.Vx < 0 or self.ego_x <0 or self.ego_x >13000 or abs(self.car_info.Vx - self.v_target)>5
        done = bool(done)

        # -----------------
        self.steps += 1
        # if self.steps >= self.max_episode_steps:
        #     done = True
        # ---------------
        weight_fuel = 1
        weight_tracking = 1e-3
        weight_control = 1e3
        #
        if not done:
            reward = -(weight_fuel * self.car_info.Qfuel + weight_tracking * ((self.ego_vel-self.v_target)**2) + weight_control * ((ades - self.ades_k)**2))
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward  = -(weight_fuel * self.car_info.Qfuel + weight_tracking * ((self.ego_vel-self.v_target)**2) + weight_control * ((ades - self.ades_k)**2))
        else:
            if self.steps_beyond_done == 0:
                gym.logger.warn("""
You are calling 'step()' even though this environment has already returned
done = True. You should always call 'reset()' once you receive 'done = True'
Any further steps are undefined behavior.
                """)
            self.steps_beyond_done += 1
            reward = -10000.0
        self.ades_k = ades

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = [0.0]+[0.0]*self.Np
        self.steps_beyond_done = None
        self.steps = 0
        return np.array(self.state)

    def render(self, mode='human'):
        pass


def env_creator(**kwargs):
    return TimeLimit(PythPCCCarCVTModel(**kwargs), 20000)

if __name__ == "__main__":
    env = PythPCCCarCVTModel()
    for i in range(10):
        next_state, reward, done, _ = env.step(4.0)
        print(next_state)

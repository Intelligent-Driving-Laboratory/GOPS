#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab

import copy
import math
import os
import random
import sys
from collections import defaultdict
from math import fabs, cos, sin, pi

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import sumolib
from sumolib import checkBinary
import traci
from traci.exceptions import FatalTraCIError
from gops.env.resources.intersection.endtoend_env_utils import (
    shift_and_rotate_coordination,
    _convert_car_coord_to_sumo_coord,
    _convert_sumo_coord_to_car_coord,
    xy2_edgeID_lane,
    SUMOCFG_DIR,
    TASK2ROUTEID,
)

SUMO_BINARY = checkBinary("sumo")
SIM_PERIOD = 1.0 / 10


class Traffic(object):
    def __init__(self, step_length, mode, init_n_ego_dict, training_task="left"):  # mode 'display' or 'training'
        self.random_traffic = None
        self.sim_time = 0
        self.n_ego_vehicles = defaultdict(list)
        self.step_length = step_length
        self.step_time_str = str(float(step_length) / 1000)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.v_light = None
        self.n_ego_dict = init_n_ego_dict
        # dict(DL1=dict(x=1.875, y=-30, v=3, a=90, l=4.8, w=2.2),
        #      UR1=dict(x=-1.875, y=30, v=3, a=-90, l=4.8, w=2.2),
        #      DR1=dict(x=5.625, y=-30, v=3, a=90, l=4.8, w=2.2),
        #      RU1=dict(x=5.625, y=-30, v=3, a=90, l=4.8, w=2.2))

        self.mode = mode
        self.training_light_phase = 0
        self.training_task = training_task
        self.ego_route = TASK2ROUTEID[self.training_task]
        if training_task == "right":
            if random.random() > 0.5:
                self.training_light_phase = 2

        try:
            traci.start(
                [
                    SUMO_BINARY,
                    "-c",
                    SUMOCFG_DIR,
                    "--step-length",
                    self.step_time_str,
                    # "--lateral-resolution", "3.5",
                    "--random",
                    # "--start",
                    # "--quit-on-end",
                    "--no-warnings",
                    "--no-step-log",
                    # '--seed', str(int(seed))
                ],
                numRetries=5,
            )  # '--seed', str(int(seed))
        except FatalTraCIError:
            print("Retry by other port")
            port = sumolib.miscutils.getFreeSocketPort()
            traci.start(
                [
                    SUMO_BINARY,
                    "-c",
                    SUMOCFG_DIR,
                    "--step-length",
                    self.step_time_str,
                    "--lateral-resolution",
                    "3.5",
                    "--random",
                    # "--start",
                    # "--quit-on-end",
                    "--no-warnings",
                    "--no-step-log",
                    # '--seed', str(int(seed))
                ],
                port=port,
                numRetries=5,
            )  # '--seed', str(int(seed))

        # traci.vehicle.subscribeContext('collector',
        #                                traci.constants.CMD_GET_VEHICLE_VARIABLE,
        #                                999999, [traci.constants.VAR_POSITION,
        #                                         traci.constants.VAR_LENGTH,
        #                                         traci.constants.VAR_WIDTH,
        #                                         traci.constants.VAR_ANGLE,
        #                                         traci.constants.VAR_SIGNALS,
        #                                         traci.constants.VAR_SPEED,
        #                                         # traci.constants.VAR_TYPE,
        #                                         # traci.constants.VAR_EMERGENCY_DECEL,
        #                                         # traci.constants.VAR_LANE_INDEX,
        #                                         # traci.constants.VAR_LANEPOSITION,
        #                                         traci.constants.VAR_EDGES,
        #                                         # traci.constants.VAR_ROUTE_INDEX
        #                                         ],
        #                                0, 2147483647)

        # traci.person.subscribeContext('00',
        #                                traci.constants.CMD_GET_PERSON_VARIABLE,
        #                                999999, [traci.constants.VAR_POSITION,
        #                                         traci.constants.VAR_LENGTH,
        #                                         traci.constants.VAR_WIDTH,
        #                                         traci.constants.VAR_ANGLE,
        #                                         # traci.constants.VAR_SIGNALS,
        #                                         traci.constants.VAR_SPEED,
        #                                         # traci.constants.VAR_TYPE,
        #                                         # traci.constants.VAR_EMERGENCY_DECEL,
        #                                         # traci.constants.VAR_LANE_INDEX,
        #                                         # traci.constants.VAR_LANEPOSITION,
        #                                         traci.constants.VAR_EDGES,
        #                                         # traci.constants.VAR_ROUTE_INDEX
        #                                         ],
        #                                0, 2147483647)

        traci.junction.subscribeContext(
            objectID="a3",
            domain=traci.constants.CMD_GET_VEHICLE_VARIABLE,
            dist=10000.0,
            varIDs=[
                traci.constants.VAR_POSITION,
                traci.constants.VAR_LENGTH,
                traci.constants.VAR_WIDTH,
                traci.constants.VAR_ANGLE,
                traci.constants.VAR_SIGNALS,
                traci.constants.VAR_SPEED,
                traci.constants.VAR_TYPE,
                # traci.constants.VAR_EMERGENCY_DECEL,
                # traci.constants.VAR_LANE_INDEX,
                # traci.constants.VAR_LANEPOSITION,
                # traci.constants.VAR_EDGES,
                # traci.constants.VAR_ROAD_ID,
                traci.constants.VAR_EDGES,
                # traci.constants.VAR_NEXT_EDGE,
                # traci.constants.VAR_ROUTE_INDEX
            ],
            begin=0.0,
            end=2147483647.0,
        )

        traci.junction.subscribeContext(
            objectID="a4",
            domain=traci.constants.CMD_GET_PERSON_VARIABLE,
            dist=10000.0,
            varIDs=[
                traci.constants.VAR_POSITION,
                traci.constants.VAR_LENGTH,
                traci.constants.VAR_WIDTH,
                traci.constants.VAR_ANGLE,
                # traci.constants.VAR_SIGNALS,
                traci.constants.VAR_SPEED,
                traci.constants.VAR_TYPE,
                # traci.constants.VAR_EMERGENCY_DECEL,
                # traci.constants.VAR_LANE_INDEX,
                # traci.constants.VAR_LANEPOSITION,
                # traci.constants.VAR_EDGES,
                traci.constants.VAR_ROAD_ID,
                # traci.constants.VAR_NEXT_EDGE,
                # traci.constants.VAR_ROUTE_ID,
                # traci.constants.VAR_ROUTE_INDEX
            ],
            begin=0.0,
            end=2147483647.0,
        )

        while traci.simulation.getTime() < 100:  # turn right
            if traci.simulation.getTime() < 80:
                traci.trafficlight.setPhase("0", 2)
            else:
                traci.trafficlight.setPhase("0", 0)

            # if self.mode == "training":
            #     traci.trafficlight.setPhase('0', self.training_light_phase)
            traci.simulationStep()

    def __del__(self):
        traci.close()

    def add_self_car(self, n_ego_dict, with_delete=True):
        for egoID, ego_dict in n_ego_dict.items():
            ego_v_x = ego_dict["v_x"]
            ego_v_y = ego_dict["v_y"]
            ego_l = ego_dict["l"]
            ego_x = ego_dict["x"]
            ego_y = ego_dict["y"]
            ego_phi = ego_dict["phi"]
            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(ego_x, ego_y, ego_phi, ego_l)
            edgeID, lane = xy2_edgeID_lane(ego_x, ego_y)
            if with_delete:
                try:
                    traci.vehicle.remove(egoID)
                except traci.exceptions.TraCIException:
                    print("Don't worry, it's been handled well")
                traci.simulationStep()
                traci.vehicle.addLegacy(
                    vehID=egoID,
                    routeID=ego_dict["routeID"],
                    # depart=0, pos=20, lane=lane, speed=ego_dict['v_x'],
                    typeID="self_car",
                )
            traci.vehicle.moveToXY(egoID, edgeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keepRoute=1)
            traci.vehicle.setLength(egoID, ego_dict["l"])
            traci.vehicle.setWidth(egoID, ego_dict["w"])
            traci.vehicle.setSpeed(egoID, math.sqrt(ego_v_x**2 + ego_v_y**2))

    def generate_random_traffic(self):
        # random_traffic = traci.vehicle.getContextSubscriptionResults('collector')
        # random_traffic = traci.person.getContextSubscriptionResults('00')
        random_traffic_01 = traci.junction.getContextSubscriptionResults("a3")
        random_traffic_02 = traci.junction.getContextSubscriptionResults("a4")
        random_traffic = dict(random_traffic_01, **random_traffic_02)
        # print("randomtraffic",random_traffic)
        random_traffic = copy.deepcopy(random_traffic)

        for ego_id in self.n_ego_dict.keys():
            if ego_id in random_traffic:
                del random_traffic[ego_id]

        return random_traffic

    def init_traffic(self, init_n_ego_dict):
        self.sim_time = 0
        self.n_ego_vehicles = defaultdict(list)
        self.collision_flag = False
        self.n_ego_collision_flag = {}
        self.collision_ego_id = None
        self.v_light = None
        self.training_light_phase = random.randint(0, 1)
        if self.training_task == "right":
            if random.random() > 0.5:
                self.training_light_phase = 2
        self.n_ego_dict = init_n_ego_dict
        traci.trafficlight.setPhase("0", self.training_light_phase)
        self.add_self_car(init_n_ego_dict)
        traci.simulationStep()
        random_traffic = self.generate_random_traffic()
        self.add_self_car(init_n_ego_dict, with_delete=False)

        # move ego to the given position and remove conflict cars
        for egoID, ego_dict in self.n_ego_dict.items():
            ego_x, ego_y, ego_v_x, ego_v_y, ego_phi, ego_l, ego_w = (
                ego_dict["x"],
                ego_dict["y"],
                ego_dict["v_x"],
                ego_dict["v_y"],
                ego_dict["phi"],
                ego_dict["l"],
                ego_dict["w"],
            )
            for veh in random_traffic:
                x_in_sumo, y_in_sumo = random_traffic[veh][traci.constants.VAR_POSITION]
                a_in_sumo = random_traffic[veh][traci.constants.VAR_ANGLE]
                veh_l = random_traffic[veh][traci.constants.VAR_LENGTH]
                veh_v = random_traffic[veh][traci.constants.VAR_SPEED]
                veh_type = random_traffic[veh][traci.constants.VAR_TYPE]
                # veh_sig = random_traffic[veh][traci.constants.VAR_SIGNALS]
                # 10: left and brake 9: right and brake 1: right 8: brake 0: no signal 2: left

                x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, veh_l)
                x_in_ego_coord, y_in_ego_coord, a_in_ego_coord = shift_and_rotate_coordination(
                    x, y, a, ego_x, ego_y, ego_phi
                )
                ego_x_in_veh_coord, ego_y_in_veh_coord, ego_a_in_veh_coord = shift_and_rotate_coordination(
                    0, 0, 0, x_in_ego_coord, y_in_ego_coord, a_in_ego_coord
                )
                if (
                    -5 < x_in_ego_coord < 1 * (ego_v_x) + ego_l / 2.0 + veh_l / 2.0 + 2 and abs(y_in_ego_coord) < 3
                ) or (
                    -5 < ego_x_in_veh_coord < 1 * (veh_v) + ego_l / 2.0 + veh_l / 2.0 + 2
                    and abs(ego_y_in_veh_coord) < 3
                ):
                    if veh_type == "DEFAULT_PEDTYPE":
                        traci.person.removeStages(veh)
                    else:
                        traci.vehicle.remove(veh)

                    # traci.vehicle.remove(vehID=veh)
                # if 0<x_in_sumo<3.5 and -22<y_in_sumo<-15:# and veh_sig!=1 and veh_sig!=9:
                #     traci.vehicle.moveToXY(veh, '4o', 1, -80, 1.85, 180,2)
                #     traci.vehicle.remove(vehID=veh)

    def _get_vehicles(self):
        self.n_ego_vehicles = defaultdict(list)
        # veh_infos = traci.vehicle.getContextSubscriptionResults('collector')
        # veh_infos = traci.person.getContextSubscriptionResults('00')
        # print("22222")
        veh_infos_01 = traci.junction.getContextSubscriptionResults("a3")
        veh_infos_02 = traci.junction.getContextSubscriptionResults("a4")
        veh_infos = dict(veh_infos_01, **veh_infos_02)
        # veh_infos = traci.junction.getContextSubscriptionResults('a4')
        # print("订阅器输出信息：", len(veh_infos), veh_infos)
        for egoID in self.n_ego_dict.keys():
            veh_info_dict = copy.deepcopy(veh_infos)
            for i, veh in enumerate(veh_info_dict):
                if veh != egoID:
                    length = veh_info_dict[veh][traci.constants.VAR_LENGTH]
                    width = veh_info_dict[veh][traci.constants.VAR_WIDTH]
                    type = veh_info_dict[veh][traci.constants.VAR_TYPE]
                    if type == "DEFAULT_PEDTYPE":
                        route = "0 0"
                    else:
                        route = veh_info_dict[veh][traci.constants.VAR_EDGES]
                    if type == "DEFAULT_PEDTYPE":
                        road = veh_info_dict[veh][traci.constants.VAR_ROAD_ID]
                    else:
                        road = "0"
                    if route[0] == "4i":
                        continue
                    x_in_sumo, y_in_sumo = veh_info_dict[veh][traci.constants.VAR_POSITION]
                    a_in_sumo = veh_info_dict[veh][traci.constants.VAR_ANGLE]
                    # transfer x,y,a in car coord
                    x, y, a = _convert_sumo_coord_to_car_coord(x_in_sumo, y_in_sumo, a_in_sumo, length)
                    v = veh_info_dict[veh][traci.constants.VAR_SPEED]
                    self.n_ego_vehicles[egoID].append(
                        dict(type=type, x=x, y=y, v=v, phi=a, l=length, w=width, route=route, road=road)
                    )

    def _get_traffic_light(self):
        self.v_light = traci.trafficlight.getPhase("0")

    def sim_step(self):
        self.sim_time += SIM_PERIOD
        if self.mode == "training":
            traci.trafficlight.setPhase("0", self.training_light_phase)
        # else:
        #     if self.sim_time < 5.:
        #         traci.trafficlight.setPhase('0', 2)
        #     elif self.sim_time < 5.+3.:
        #         traci.trafficlight.setPhase('0', 1)
        #     else:
        #         traci.trafficlight.setPhase('0', 0)
        traci.simulationStep()
        self._get_vehicles()
        self._get_traffic_light()
        self.collision_check()
        for egoID, collision_flag in self.n_ego_collision_flag.items():
            if collision_flag:
                self.collision_flag = True
                self.collision_ego_id = egoID

    def set_own_car(self, n_ego_dict_):
        assert len(self.n_ego_dict) == len(n_ego_dict_)
        for egoID in self.n_ego_dict.keys():
            self.n_ego_dict[egoID]["v_x"] = ego_v_x = n_ego_dict_[egoID]["v_x"]
            self.n_ego_dict[egoID]["v_y"] = ego_v_y = n_ego_dict_[egoID]["v_y"]
            self.n_ego_dict[egoID]["r"] = ego_r = n_ego_dict_[egoID]["r"]
            self.n_ego_dict[egoID]["x"] = ego_x = n_ego_dict_[egoID]["x"]
            self.n_ego_dict[egoID]["y"] = ego_y = n_ego_dict_[egoID]["y"]
            self.n_ego_dict[egoID]["phi"] = ego_phi = n_ego_dict_[egoID]["phi"]

            ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo = _convert_car_coord_to_sumo_coord(
                ego_x, ego_y, ego_phi, self.n_ego_dict[egoID]["l"]
            )
            egdeID, lane = xy2_edgeID_lane(ego_x, ego_y)
            keeproute = 2
            # if self.training_task == 'left':  #
            #     keeproute = 2 if ego_x > 0 and ego_y > -7 else 1
            try:
                traci.vehicle.moveToXY(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
            except traci.exceptions.TraCIException:
                print(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
                traci.vehicle.moveToXY(egoID, egdeID, lane, ego_x_in_sumo, ego_y_in_sumo, ego_a_in_sumo, keeproute)
            traci.vehicle.setSpeed(egoID, math.sqrt(ego_v_x**2 + ego_v_y**2))

    def collision_check(self):  # True: collision
        flag_dict = dict()
        for egoID, list_of_veh_dict in self.n_ego_vehicles.items():
            ego_x = self.n_ego_dict[egoID]["x"]
            ego_y = self.n_ego_dict[egoID]["y"]
            ego_phi = self.n_ego_dict[egoID]["phi"]
            ego_l = self.n_ego_dict[egoID]["l"]
            ego_w = self.n_ego_dict[egoID]["w"]
            ego_lw = (ego_l - ego_w) / 2
            ego_x0 = ego_x + cos(ego_phi / 180 * pi) * ego_lw
            ego_y0 = ego_y + sin(ego_phi / 180 * pi) * ego_lw
            ego_x1 = ego_x - cos(ego_phi / 180 * pi) * ego_lw
            ego_y1 = ego_y - sin(ego_phi / 180 * pi) * ego_lw
            flag_dict[egoID] = False

            for veh in list_of_veh_dict:
                if fabs(veh["x"] - ego_x) < 10 and fabs(veh["y"] - ego_y) < 10:
                    surrounding_lw = (veh["l"] - veh["w"]) / 2
                    surrounding_x0 = veh["x"] + cos(veh["phi"] / 180 * pi) * surrounding_lw
                    surrounding_y0 = veh["y"] + sin(veh["phi"] / 180 * pi) * surrounding_lw
                    surrounding_x1 = veh["x"] - cos(veh["phi"] / 180 * pi) * surrounding_lw
                    surrounding_y1 = veh["y"] - sin(veh["phi"] / 180 * pi) * surrounding_lw
                    collision_check_dis = ((veh["w"] + ego_w) / 2 + 0.5) ** 2
                    if (ego_x0 - surrounding_x0) ** 2 + (ego_y0 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x0 - surrounding_x1) ** 2 + (ego_y0 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x1) ** 2 + (ego_y1 - surrounding_y1) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True
                    elif (ego_x1 - surrounding_x0) ** 2 + (ego_y1 - surrounding_y0) ** 2 < collision_check_dis:
                        flag_dict[egoID] = True

        self.n_ego_collision_flag = flag_dict


def test_traffic():
    import numpy as np
    from dynamics_and_models import ReferencePath

    def _reset_init_state():
        ref_path = ReferencePath("straight")
        random_index = int(np.random.random() * (900 + 500)) + 700
        x, y, phi = ref_path.indexs2points(random_index)
        v = 8 * np.random.random()
        return dict(
            ego=dict(
                v_x=v,
                v_y=0,
                r=0,
                x=x.numpy(),
                y=y.numpy(),
                phi=phi.numpy(),
                l=4.8,
                w=2.2,
                routeID="du",
            )
        )

    init_state = dict(
        ego=dict(
            v_x=8.0,
            v_y=0,
            r=0,
            x=-30,
            y=1.5,
            phi=180,
            l=4.8,
            w=2.2,
            routeID="dl",
        )
    )
    # init_state = _reset_init_state()
    traffic = Traffic(100.0, mode="training", init_n_ego_dict=init_state, training_task="left")
    traffic.init_traffic(init_state)
    traffic.sim_step()
    for i in range(100000000):
        # for j in range(50):
        # traffic.set_own_car(init_state)
        # traffic.sim_step()
        # init_state = _reset_init_state()
        # traffic.init_traffic(init_state)
        traffic.sim_step()


if __name__ == "__main__":
    test_traffic()

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Crossing Road End to End Utils
#  Update Date: 2021-05-55, Yang Guan: create crossing road end to end utils



import math
import os
from collections import OrderedDict

L, W = 4.8, 2.0
L_BIKE, W_BIKE = 2.0, 0.48
LANE_WIDTH = 3.75
BIKE_LANE_WIDTH = 2.0
PERSON_LANE_WIDTH = 2.0
LANE_NUMBER = 3
CROSSROAD_SIZE = 50
EXPECTED_V = 8.
dirname = os.path.dirname(__file__)
SUMOCFG_DIR = dirname + "/sumo_files/cross.sumocfg"
VEHICLE_MODE_DICT = dict(left=OrderedDict(dl=2, du=2, ud=2, ul=2),
                         straight=OrderedDict(dl=1, du=2, ru=2, ur=2),
                         right=OrderedDict(dr=1, ur=2, lr=2))
BIKE_MODE_DICT = dict(left=OrderedDict(ud_b=2),
                         straight=OrderedDict(du_b=4),
                         right=OrderedDict(du_b=2, lr_b=0))
PERSON_MODE_DICT = dict(left=OrderedDict(c3=4),
                         straight=OrderedDict(c2=0),
                         right=OrderedDict(c1=4, c2=0))


def dict2flat(inp):
    out = []
    for key, val in inp.items():
        out.extend([key]*val)
    return out


def dict2num(inp):
    out = 0
    for _, val in inp.items():
        out += val
    return out


VEH_NUM = dict(left=dict2num(VEHICLE_MODE_DICT['left']),
               straight=dict2num(VEHICLE_MODE_DICT['straight']),
               right=dict2num(VEHICLE_MODE_DICT['right']))

BIKE_NUM = dict(left=dict2num(BIKE_MODE_DICT['left']),
                straight=dict2num(BIKE_MODE_DICT['straight']),
                right=dict2num(BIKE_MODE_DICT['right']))

PERSON_NUM = dict(left=dict2num(PERSON_MODE_DICT['left']),
                  straight=dict2num(PERSON_MODE_DICT['straight']),
                  right=dict2num(PERSON_MODE_DICT['right']))

VEHICLE_MODE_LIST = dict(left=dict2flat(VEHICLE_MODE_DICT['left']),
                         straight=dict2flat(VEHICLE_MODE_DICT['straight']),
                         right=dict2flat(VEHICLE_MODE_DICT['right']))
BIKE_MODE_LIST = dict(left=dict2flat(BIKE_MODE_DICT['left']),
                      straight=dict2flat(BIKE_MODE_DICT['straight']),
                      right=dict2flat(BIKE_MODE_DICT['right']))
PERSON_MODE_LIST = dict(left=dict2flat(PERSON_MODE_DICT['left']),
                        straight=dict2flat(PERSON_MODE_DICT['straight']),
                        right=dict2flat(PERSON_MODE_DICT['right']))

# Things related to lane number: static path generation (which further influences obs initialization),
# observation formulation (especially other vehicles selection and number), rewards formulation
# other vehicle prediction
# feasibility judgement
# the sumo files, obviously,
# the render func,
# it is hard to unify them using one set of code, better be a case-by-case setting.

ROUTE2MODE = {('1o', '2i'): 'dr', ('1o', '3i'): 'du', ('1o', '4i'): 'dl',
              ('2o', '1i'): 'rd', ('2o', '3i'): 'ru', ('2o', '4i'): 'rl',
              ('3o', '1i'): 'ud', ('3o', '2i'): 'ur', ('3o', '4i'): 'ul',
              ('4o', '1i'): 'ld', ('4o', '2i'): 'lr', ('4o', '3i'): 'lu'}

MODE2TASK = {'dr': 'right', 'du': 'straight', 'dl': 'left',
             'rd': 'left', 'ru': 'right', 'rl': ' straight',
             'ud': 'straight', 'ur': 'left', 'ul': 'right',
             'ld': 'right', 'lr': 'straight', 'lu': 'left',
             'ud_b': 'straight', 'du_b':'straight', 'lr_b':'straight',
             'c1':'straight', 'c2':'straight', 'c3':'straight'}

TASK2ROUTEID = {'left': 'dl', 'straight': 'du', 'right': 'dr'}

MODE2ROUTE = {'dr': ('1o', '2i'), 'du': ('1o', '3i'), 'dl': ('1o', '4i'),
              'rd': ('2o', '1i'), 'ru': ('2o', '3i'), 'rl': ('2o', '4i'),
              'ud': ('3o', '1i'), 'ur': ('3o', '2i'), 'ul': ('3o', '4i'),
              'ld': ('4o', '1i'), 'lr': ('4o', '2i'), 'lu': ('4o', '3i')}


def judge_feasible(orig_x, orig_y, task):  # map dependant
    def is_in_straight_before1(orig_x, orig_y):
        return 0 < orig_x < LANE_WIDTH and orig_y <= -CROSSROAD_SIZE / 2

    def is_in_straight_before2(orig_x, orig_y):
        return LANE_WIDTH < orig_x < LANE_WIDTH * 2 and orig_y <= -CROSSROAD_SIZE / 2

    def is_in_straight_before3(orig_x, orig_y):
        return LANE_WIDTH * 2 < orig_x < LANE_WIDTH * 3 and orig_y <= -CROSSROAD_SIZE / 2

    def is_in_straight_after(orig_x, orig_y):
        return 0 < orig_x < LANE_WIDTH * LANE_NUMBER and orig_y >= CROSSROAD_SIZE / 2

    def is_in_left(orig_x, orig_y):
        return 0 < orig_y < LANE_WIDTH * LANE_NUMBER and orig_x < -CROSSROAD_SIZE / 2

    def is_in_right(orig_x, orig_y):
        return -LANE_WIDTH * LANE_NUMBER < orig_y < 0 and orig_x > CROSSROAD_SIZE / 2

    def is_in_middle(orig_x, orig_y):
        return True if -CROSSROAD_SIZE / 2 < orig_y < CROSSROAD_SIZE / 2 and -CROSSROAD_SIZE / 2 < orig_x < CROSSROAD_SIZE / 2 else False

    if task == 'left':
        return True if is_in_straight_before1(orig_x, orig_y) or is_in_left(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False
    elif task == 'straight':
        return True if is_in_straight_before2(orig_x, orig_y) or is_in_straight_after(
            orig_x, orig_y) or is_in_middle(orig_x, orig_y) else False
    else:
        assert task == 'right'
        return True if is_in_straight_before3(orig_x, orig_y) or is_in_right(orig_x, orig_y) \
                       or is_in_middle(orig_x, orig_y) else False


def shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y):
    '''
    :param orig_x: original x
    :param orig_y: original y
    :param coordi_shift_x: coordi_shift_x along x axis
    :param coordi_shift_y: coordi_shift_y along y axis
    :return: shifted_x, shifted_y
    '''
    shifted_x = orig_x - coordi_shift_x
    shifted_y = orig_y - coordi_shift_y
    return shifted_x, shifted_y


def rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d):
    """
    :param orig_x: original x
    :param orig_y: original y
    :param orig_d: original degree
    :param coordi_rotate_d: coordination rotation d, positive if anti-clockwise, unit: deg
    :return:
    transformed_x, transformed_y, transformed_d(range:(-180 deg, 180 deg])
    """

    coordi_rotate_d_in_rad = coordi_rotate_d * math.pi / 180
    transformed_x = orig_x * math.cos(coordi_rotate_d_in_rad) + orig_y * math.sin(coordi_rotate_d_in_rad)
    transformed_y = -orig_x * math.sin(coordi_rotate_d_in_rad) + orig_y * math.cos(coordi_rotate_d_in_rad)
    transformed_d = orig_d - coordi_rotate_d
    if transformed_d > 180:
        while transformed_d > 180:
            transformed_d = transformed_d - 360
    elif transformed_d <= -180:
        while transformed_d <= -180:
            transformed_d = transformed_d + 360
    else:
        transformed_d = transformed_d
    return transformed_x, transformed_y, transformed_d


def shift_and_rotate_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y = shift_coordination(orig_x, orig_y, coordi_shift_x, coordi_shift_y)
    transformed_x, transformed_y, transformed_d \
        = rotate_coordination(shift_x, shift_y, orig_d, coordi_rotate_d)
    return transformed_x, transformed_y, transformed_d


def rotate_and_shift_coordination(orig_x, orig_y, orig_d, coordi_shift_x, coordi_shift_y, coordi_rotate_d):
    shift_x, shift_y, transformed_d \
        = rotate_coordination(orig_x, orig_y, orig_d, coordi_rotate_d)
    transformed_x, transformed_y = shift_coordination(shift_x, shift_y, coordi_shift_x, coordi_shift_y)

    return transformed_x, transformed_y, transformed_d


def cal_info_in_transform_coordination(filtered_objects, x, y, rotate_d):  # rotate_d is positive if anti
    results = []
    for obj in filtered_objects:
        orig_x = obj['x']
        orig_y = obj['y']
        orig_v = obj['v']
        orig_heading = obj['phi']
        width = obj['w']
        length = obj['l']
        route = obj['route']
        shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
        trans_x, trans_y, trans_heading = rotate_coordination(shifted_x, shifted_y, orig_heading, rotate_d)
        trans_v = orig_v
        results.append({'x': trans_x,
                        'y': trans_y,
                        'v': trans_v,
                        'phi': trans_heading,
                        'w': width,
                        'l': length,
                        'route': route,})
    return results


def cal_ego_info_in_transform_coordination(ego_dynamics, x, y, rotate_d):
    orig_x, orig_y, orig_a, corner_points = ego_dynamics['x'], ego_dynamics['y'], ego_dynamics['phi'], ego_dynamics['Corner_point']
    shifted_x, shifted_y = shift_coordination(orig_x, orig_y, x, y)
    trans_x, trans_y, trans_a = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
    trans_corner_points = []
    for corner_x, corner_y in corner_points:
        shifted_x, shifted_y = shift_coordination(corner_x, corner_y, x, y)
        trans_corner_x, trans_corner_y, _ = rotate_coordination(shifted_x, shifted_y, orig_a, rotate_d)
        trans_corner_points.append((trans_corner_x, trans_corner_y))
    ego_dynamics.update(dict(x=trans_x,
                             y=trans_y,
                             phi=trans_a,
                             Corner_point=trans_corner_points))
    return ego_dynamics


def xy2_edgeID_lane(x, y):
    if y < -CROSSROAD_SIZE/2:
        edgeID = '1o'
        lane = int((LANE_NUMBER-1)-int(x/LANE_WIDTH))
    elif x < -CROSSROAD_SIZE/2:
        edgeID = '4i'
        lane = int((LANE_NUMBER-1)-int(y/LANE_WIDTH))
    elif y > CROSSROAD_SIZE/2:
        edgeID = '3i'
        lane = int((LANE_NUMBER-1)-int(x/LANE_WIDTH))
    elif x > CROSSROAD_SIZE/2:
        edgeID = '2i'
        lane = int((LANE_NUMBER-1)-int(-y/LANE_WIDTH))
    else:
        edgeID = '0'
        lane = 0
    return edgeID, lane


def _convert_car_coord_to_sumo_coord(x_in_car_coord, y_in_car_coord, a_in_car_coord, car_length):  # a in deg
    x_in_sumo_coord = x_in_car_coord + car_length / 2 * math.cos(math.radians(a_in_car_coord))
    y_in_sumo_coord = y_in_car_coord + car_length / 2 * math.sin(math.radians(a_in_car_coord))
    a_in_sumo_coord = -a_in_car_coord + 90.
    return x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord


def _convert_sumo_coord_to_car_coord(x_in_sumo_coord, y_in_sumo_coord, a_in_sumo_coord, car_length):
    a_in_car_coord = -a_in_sumo_coord + 90.
    x_in_car_coord = x_in_sumo_coord - (math.cos(a_in_car_coord / 180. * math.pi) * car_length / 2)
    y_in_car_coord = y_in_sumo_coord - (math.sin(a_in_car_coord / 180. * math.pi) * car_length / 2)
    return x_in_car_coord, y_in_car_coord, deal_with_phi(a_in_car_coord)


def deal_with_phi(phi):
    while phi > 180:
        phi -= 360
    while phi <= -180:
        phi += 360
    return phi


if __name__ == '__main__':
    pass

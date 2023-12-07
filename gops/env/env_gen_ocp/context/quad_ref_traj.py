import numpy as np
from enum import IntEnum
import configparser
config = configparser.ConfigParser()
import configparser
from enum import Enum
from gops.env.env_gen_ocp.pyth_base import ContextState, Context

class QuadType(Enum):
    ONE_D = 1
    TWO_D = 2
    THREE_D = 3

# config = configparser.ConfigParser()
# config.read('../config.ini')
# # 读取字符串并将其转换回枚举类型
# parameter = QuadType[config.get('quad_type', 'quad_type')]
# print("Parameter from config file:", parameter)

class QuadType(IntEnum):
    '''Quadrotor types numeration class.'''
    ONE_D = 1  # One-dimensional (along z) movement.
    TWO_D = 2  # Two-dimensional (in the x-z plane) movement.
    THREE_D = 3  # Three-dimensional movement.



class QuadContext(Context):
    def __init__(self,
                 offset = [1, 0],
                 *,
                 prior_prop={}, 
                 quad_type = QuadType.ONE_D,
                 rew_state_weight=1.0,
                 rew_act_weight=0.01,
                #  rew_act_weight=0.0001,
                 
                 pre_horizon: int = 10,
                 task = 'TRAJ_TRACKING'
        ) -> None:
        self.QUAD_TYPE = QuadType(quad_type)
        self.task = task
        self.ctrl_step_counter = 0
        self.MASS = prior_prop.get('M', 1.0)
        self.rew_state_weight = np.array(rew_state_weight, ndmin=1, dtype=float)
        self.rew_act_weight = np.array(rew_act_weight, ndmin=1, dtype=float)
        self.pre_horizon = pre_horizon
        self.TASK_INFO = {
        'stabilization_goal': [0, 1, 0],
        'stabilization_goal_tolerance': 0.05,
        'trajectory_type': 'circle',
        'num_cycles': 1,
        'trajectory_plane': 'zx',
        'trajectory_position_offset': [0.5, 0],
        'trajectory_scale': -0.5,
        'proj_point': [0, 0, 0.5],
        'proj_normal': [0, 1, 1],
    }
        self._get_GOAL(offset)
        self.reset()
    
    def reset(self) -> ContextState[np.ndarray]:
        self.ctrl_step_counter = 0
        ref_points = self.X_GOAL[self.ctrl_step_counter :self.ctrl_step_counter + self.pre_horizon + 1]
        self.state = ContextState(reference=ref_points)
        return self.state
    
    def step(self) -> ContextState[np.ndarray]:
        self.ctrl_step_counter += 1
        wp_idx = min(self.ctrl_step_counter, self.X_GOAL.shape[0] - 1)  
        new_ref_point = self.X_GOAL[wp_idx]
        # self.state = ContextState(np.expand_dims(self.X_GOAL[wp_idx], axis=0))
        ref_points = self.state.reference.copy()
        ref_points[:-1] = ref_points[1:]
        ref_points[-1] = new_ref_point
        self.state.reference = ref_points
        return self.state
    
    def get_zero_state(self) -> ContextState[np.ndarray]:
        return ContextState(
            reference=np.zeros((self.pre_horizon + 1, len(self.X_GOAL[0])), dtype=np.float32))
        
    def _generate_trajectory(self,
                             traj_type='figure8',
                             traj_length=10.0,
                             num_cycles=1,
                             traj_plane='xy',
                             position_offset=np.array([0, 0]),
                             scaling=1.0,
                             sample_time=0.01):
        '''Generates a 2D trajectory.
        Args:
            traj_type (str, optional): The type of trajectory (circle, square, figure8).
            traj_length (float, optional): The length of the trajectory in seconds.
            num_cycles (int, optional): The number of cycles within the length.
            traj_plane (str, optional): The plane of the trajectory (e.g. 'xz').
            position_offset (ndarray, optional): An initial position offset in the plane.
            scaling (float, optional): Scaling factor for the trajectory.
            sample_time (float, optional): The sampling timestep of the trajectory.
        Returns:
            ndarray: The positions in x, y, z of the trajectory sampled for its entire duration.
            ndarray: The velocities in x, y, z of the trajectory sampled for its entire duration.
            ndarray: The scalar speed of the trajectory sampled for its entire duration.
        '''
        # Get trajectory type.
        valid_traj_type = ['circle', 'square', 'figure8']
        if traj_type not in valid_traj_type:
            raise ValueError('Trajectory type should be one of [circle, square, figure8].')
        traj_period = traj_length / num_cycles
        direction_list = ['x', 'y', 'z']
        # Get coordinates indexes.
        if traj_plane[0] in direction_list and traj_plane[
                1] in direction_list and traj_plane[0] != traj_plane[1]:
            coord_index_a = direction_list.index(traj_plane[0])
            coord_index_b = direction_list.index(traj_plane[1])
        else:
            raise ValueError('Trajectory plane should be in form of ab, where a and b can be {x, y, z}.')
        # Generate time stamps.
        times = np.arange(0, traj_length + sample_time, sample_time)  # sample time added to make reference one step longer than traj_length
        pos_ref_traj = np.zeros((len(times), 3))
        vel_ref_traj = np.zeros((len(times), 3))
        speed_traj = np.zeros((len(times), 1))
        # Compute trajectory points.
        for t in enumerate(times):
            pos_ref_traj[t[0]], vel_ref_traj[t[0]] = self._get_coordinates(t[1],
                                                                           traj_type,
                                                                           traj_period,
                                                                           coord_index_a,
                                                                           coord_index_b,
                                                                           position_offset[0],
                                                                           position_offset[1],
                                                                           scaling)
            speed_traj[t[0]] = np.linalg.norm(vel_ref_traj[t[0]])
        return pos_ref_traj, vel_ref_traj, speed_traj
    
    
    def _get_coordinates(self,
                         t,
                         traj_type,
                         traj_period,
                         coord_index_a,
                         coord_index_b,
                         position_offset_a,
                         position_offset_b,
                         scaling
                         ):
        '''Computes the coordinates of a specified trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_type (str, optional): The type of trajectory (circle, square, figure8).
            traj_period (float): The period of the trajectory in seconds.
            coord_index_a (int): The index of the first coordinate of the trajectory plane.
            coord_index_b (int): The index of the second coordinate of the trajectory plane.
            position_offset_a (float): The offset in the first coordinate of the trajectory plane.
            position_offset_b (float): The offset in the second coordinate of the trajectory plane.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            pos_ref (ndarray): The position in x, y, z, at time t.
            vel_ref (ndarray): The velocity in x, y, z, at time t.
        '''

        # Get coordinates for the trajectory chosen.
        if traj_type == 'figure8':
            coords_a, coords_b, coords_a_dot, coords_b_dot = self._figure8(
                t, traj_period, scaling)
        elif traj_type == 'circle':
            coords_a, coords_b, coords_a_dot, coords_b_dot = self._circle(
                t, traj_period, scaling)
        elif traj_type == 'square':
            coords_a, coords_b, coords_a_dot, coords_b_dot = self._square(
                t, traj_period, scaling)
        # Initialize position and velocity references.
        pos_ref = np.zeros((3,))
        vel_ref = np.zeros((3,))
        # Set position and velocity references based on the plane of the trajectory chosen.
        pos_ref[coord_index_a] = coords_a + position_offset_a
        vel_ref[coord_index_a] = coords_a_dot
        pos_ref[coord_index_b] = coords_b + position_offset_b
        vel_ref[coord_index_b] = coords_b_dot
        return pos_ref, vel_ref

    def _get_GOAL(self,offset):
        # Create X_GOAL and U_GOAL references for the assigned task.
        self.action_dim = 1
        self.GRAVITY_ACC = 9.8
        self.EPISODE_LEN_SEC = 20
        self.CTRL_FREQ = 100
        self.CTRL_TIMESTEP = 0.1
        self.CTRL_STEPS = self.EPISODE_LEN_SEC *  self.CTRL_FREQ
        self.U_GOAL = np.ones(self.action_dim) * self.MASS * self.GRAVITY_ACC / self.action_dim
        if self.task == 'STABILIZATION':
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.X_GOAL = np.expand_dims(np.hstack(
                    [self.TASK_INFO['stabilization_goal'][1],
                     0.0]),axis=0)  # x = {z, z_dot}.
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0, 0.0, 0.0
                ])  # x = {x, x_dot, z, z_dot, theta, theta_dot}.
            elif self.QUAD_TYPE == QuadType.THREE_D:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0,
                    self.TASK_INFO['stabilization_goal'][2], 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ])  # x = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
        
        elif self.task == 'TRAJ_TRACKING':
            POS_REF, VEL_REF, _ = self._generate_trajectory(traj_type=self.TASK_INFO['trajectory_type'],
                                                            traj_length=self.EPISODE_LEN_SEC,
                                                            num_cycles=self.TASK_INFO['num_cycles'],
                                                            traj_plane=self.TASK_INFO['trajectory_plane'],
                                                            position_offset=offset,
                                                            scaling=self.TASK_INFO['trajectory_scale'],
                                                            sample_time=self.CTRL_TIMESTEP
                                                            )  # Each of the 3 returned values is of shape (Ctrl timesteps, 3)
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 2],  # z
                    VEL_REF[:, 2]  # z_dot
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0],  # x
                    VEL_REF[:, 0],  # x_dot
                    POS_REF[:, 2],  # z
                    VEL_REF[:, 2],  # z_dot
                    np.zeros(POS_REF.shape[0]),  # zeros
                    np.zeros(VEL_REF.shape[0])
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.THREE_D:
                # Additional transformation of the originally planar trajectory.
               
                def transform_trajectory(pos, vel, trans_info={}):
                    '''Makes 2D reference trajectory into a 3D one.
                    Args:
                        pos: position in the reference trajectory, with shape (T,3).
                        vel: velocity in the reference trajectory, with shape (T,3).
                    '''
                    # Shape (4,4) with augmented last dim (always 1).
                    M = projection_matrix(trans_info['point'], trans_info['normal'])
                    # Position.
                    aug_pos = np.concatenate([pos, np.ones((pos.shape[0], 1))], -1)  # (T,4)
                    trans_pos = np.matmul(aug_pos, M.transpose())[:, :3]  # (T,3)
                    # Velocity (transfomration is linear, direclty multiply for derivatives).
                    aug_vel = np.concatenate([vel, np.ones((vel.shape[0], 1))], -1)  # (T,4)
                    trans_vel = np.matmul(aug_vel, M.transpose())[:, :3]  # (T,3)
                    return trans_pos, trans_vel
                
                def projection_matrix(point, normal, direction=None, perspective=None, pseudo=False):
                    M = np.identity(4)
                    point = np.array(point[:3], dtype=np.float64, copy=False)
                    normal = unit_vector(normal[:3])
                    if perspective is not None:
                        # perspective projection
                        perspective = np.array(perspective[:3], dtype=np.float64, copy=False)
                        M[0, 0] = M[1, 1] = M[2, 2] = np.dot(perspective - point, normal)
                        M[:3, :3] -= np.outer(perspective, normal)
                        if pseudo:
                            # preserve relative depth
                            M[:3, :3] -= np.outer(normal, normal)
                            M[:3, 3] = np.dot(point, normal) * (perspective + normal)
                        else:
                            M[:3, 3] = np.dot(point, normal) * perspective
                        M[3, :3] = -normal
                        M[3, 3] = np.dot(perspective, normal)
                    elif direction is not None:
                        # parallel projection
                        direction = np.array(direction[:3], dtype=np.float64, copy=False)
                        scale = np.dot(direction, normal)
                        M[:3, :3] -= np.outer(direction, normal) / scale
                        M[:3, 3] = direction * (np.dot(point, normal) / scale)
                    else:
                        # orthogonal projection
                        M[:3, :3] -= np.outer(normal, normal)
                        M[:3, 3] = np.dot(point, normal) * normal
                    return M
                def unit_vector(data, axis=None, out=None):
                    if out is None:
                        data = np.array(data, dtype=np.float64, copy=True)
                        if data.ndim == 1:
                            data /= np.sqrt(np.dot(data, data))
                            return data
                    else:
                        if out is not data:
                            out[:] = np.array(data, copy=False)
                        data = out
                    length = np.atleast_1d(np.sum(data * data, axis))
                    np.sqrt(length, length)
                    if axis is not None:
                        length = np.expand_dims(length, axis)
                    data /= length
                    if out is None:
                        return data
                POS_REF_TRANS, VEL_REF_TRANS = transform_trajectory(
                    POS_REF, VEL_REF, trans_info={
                        'point': self.TASK_INFO['proj_point'],
                        'normal': self.TASK_INFO['proj_normal'],
                    })
                
                self.X_GOAL = np.vstack([
                    POS_REF_TRANS[:, 0],  # x
                    VEL_REF_TRANS[:, 0],  # x_dot
                    POS_REF_TRANS[:, 1],  # y
                    VEL_REF_TRANS[:, 1],  # y_dot
                    POS_REF_TRANS[:, 2],  # z
                    VEL_REF_TRANS[:, 2],  # z_dot
                    np.zeros(POS_REF_TRANS.shape[0]),  # zeros
                    np.zeros(POS_REF_TRANS.shape[0]),
                    np.zeros(POS_REF_TRANS.shape[0]),
                    np.zeros(VEL_REF_TRANS.shape[0]),
                    np.zeros(VEL_REF_TRANS.shape[0]),
                    np.zeros(VEL_REF_TRANS.shape[0])
                ]).transpose()

    def _figure8(self,
                 t,
                 traj_period,
                 scaling
                 ):
        '''Computes the coordinates of a figure8 trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_period (float): The period of the trajectory in seconds.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            coords_a (float): The position in the first coordinate.
            coords_b (float): The position in the second coordinate.
            coords_a_dot (float): The velocity in the first coordinate.
            coords_b_dot (float): The velocity in the second coordinate.
        '''

        traj_freq = 2.0 * np.pi / traj_period
        coords_a = scaling * np.sin(traj_freq * t)
        coords_b = scaling * np.sin(traj_freq * t) * np.cos(traj_freq * t)
        coords_a_dot = scaling * traj_freq * np.cos(traj_freq * t)
        coords_b_dot = scaling * traj_freq * (np.cos(traj_freq * t)**2 - np.sin(traj_freq * t)**2)
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def _circle(self,
                t,
                traj_period,
                scaling
                ):
        '''Computes the coordinates of a circle trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_period (float): The period of the trajectory in seconds.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            coords_a (float): The position in the first coordinate.
            coords_b (float): The position in the second coordinate.
            coords_a_dot (float): The velocity in the first coordinate.
            coords_b_dot (float): The velocity in the second coordinate.
        '''

        traj_freq = 2.0 * np.pi / traj_period
        coords_a = scaling * np.cos(traj_freq * t)
        coords_b = scaling * np.sin(traj_freq * t)
        coords_a_dot = -scaling * traj_freq * np.sin(traj_freq * t)
        coords_b_dot = scaling * traj_freq * np.cos(traj_freq * t)
        return coords_a, coords_b, coords_a_dot, coords_b_dot

    def _square(self,
                t,
                traj_period,
                scaling
                ):
        '''Computes the coordinates of a square trajectory at time t.

        Args:
            t (float): The time at which we want to sample one trajectory point.
            traj_period (float): The period of the trajectory in seconds.
            scaling (float, optional): Scaling factor for the trajectory.

        Returns:
            coords_a (float): The position in the first coordinate.
            coords_b (float): The position in the second coordinate.
            coords_a_dot (float): The velocity in the first coordinate.
            coords_b_dot (float): The velocity in the second coordinate.
        '''

        # Compute time for each segment to complete.
        segment_period = traj_period / 4.0
        traverse_speed = scaling / segment_period
        # Compute time for the cycle.
        cycle_time = t % traj_period
        # Check time along the current segment and ratio of completion.
        segment_time = cycle_time % segment_period
        # Check current segment index.
        segment_index = int(np.floor(cycle_time / segment_period))
        # Position along segment
        segment_position = traverse_speed * segment_time
        if segment_index == 0:
            # Moving up along second axis from (0, 0).
            coords_a = 0.0
            coords_b = segment_position
            coords_a_dot = 0.0
            coords_b_dot = traverse_speed
        elif segment_index == 1:
            # Moving left along first axis from (0, 1).
            coords_a = -segment_position
            coords_b = scaling
            coords_a_dot = -traverse_speed
            coords_b_dot = 0.0
        elif segment_index == 2:
            # Moving down along second axis from (-1, 1).
            coords_a = -scaling
            coords_b = scaling - segment_position
            coords_a_dot = 0.0
            coords_b_dot = -traverse_speed
        elif segment_index == 3:
            # Moving right along second axis from (-1, 0).
            coords_a = -scaling + segment_position
            coords_b = 0.0
            coords_a_dot = traverse_speed
            coords_b_dot = 0.0
        return coords_a, coords_b, coords_a_dot, coords_b_dot
    
if __name__ == '__main__':
    quad = QuadContext()
    quad.step()
    quad.reset()
from __future__ import annotations
import vehicle3dof
import typing
import numpy
import vehicle3dof._env

__all__ = ["GymEnv", "GymEnvVec", "RawEnv", "RawEnvVec", "Veh3dofconti"]

class GymEnv:
    def __enter__(self) -> object: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> bool: ...
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, spec: _env.EnvSpec) -> None: ...
    def __repr__(self) -> str: ...
    def close(self) -> None: ...
    def render(self, mode: str = "human") -> None: ...
    def reset(self, callback: typing.Callable[[], None] = None) -> numpy.ndarray: ...
    @typing.overload
    def seed(self) -> typing.List[int]: ...
    @typing.overload
    def seed(self, seed: int) -> typing.List[int]: ...
    def step(self, action: numpy.ndarray) -> tuple: ...
    @property
    def model_class(self) -> Veh3dofconti:
        """
        :type: Veh3dofconti
        """
    @property
    def spec(self) -> _env.EnvSpec:
        """
        :type: _env.EnvSpec
        """
    @property
    def unwrapped(self) -> object:
        """
        :type: object
        """
    action_space: vehicle3dof._env.BoxFloat64  # value = Box(-3000, 3000, [3])
    metadata = {"render.modes": []}
    observation_space: vehicle3dof._env.BoxFloat64  # value = Box(-99999, 99999, [6])
    reward_range = [-99999.0, 99999.0]
    pass

class GymEnvVec:
    def __enter__(self) -> object: ...
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> bool: ...
    @typing.overload
    def __init__(self, batch_size: int) -> None: ...
    @typing.overload
    def __init__(self, batch_size: int, spec: _env.EnvSpec) -> None: ...
    def __repr__(self) -> str: ...
    def at(self, arg0: int) -> GymEnv: ...
    def close(self) -> None: ...
    def render(self, mode: str = "human") -> None: ...
    @typing.overload
    def reset(self) -> numpy.ndarray: ...
    @typing.overload
    def reset(self, indices: numpy.ndarray) -> numpy.ndarray: ...
    @typing.overload
    def reset(self, mask: numpy.ndarray) -> numpy.ndarray: ...
    @typing.overload
    def seed(self) -> typing.List[int]: ...
    @typing.overload
    def seed(self, seed: int) -> typing.List[int]: ...
    def size(self) -> int: ...
    @typing.overload
    def step(self, action: numpy.ndarray) -> tuple: ...
    @typing.overload
    def step(self, action: numpy.ndarray, indices: numpy.ndarray) -> tuple: ...
    @typing.overload
    def step(self, action: numpy.ndarray, mask: numpy.ndarray) -> tuple: ...
    @property
    def spec(self) -> _env.EnvSpec:
        """
        :type: _env.EnvSpec
        """
    @property
    def unwrapped(self) -> object:
        """
        :type: object
        """
    action_space: vehicle3dof._env.BoxFloat64  # value = Box(-3000, 3000, [3])
    metadata = {"render.modes": []}
    observation_space: vehicle3dof._env.BoxFloat64  # value = Box(-99999, 99999, [6])
    reward_range = [-99999.0, 99999.0]
    pass

class RawEnv:
    def __init__(self) -> None: ...
    def reset(self) -> Veh3dofconti.ExtY_vehicle3dof_T: ...
    @typing.overload
    def seed(self) -> typing.List[int]: ...
    @typing.overload
    def seed(self, seed: int) -> typing.List[int]: ...
    def step(
        self, action: Veh3dofconti.ExtU_vehicle3dof_T
    ) -> Veh3dofconti.ExtY_vehicle3dof_T: ...
    @property
    def model_class(self) -> Veh3dofconti:
        """
        :type: Veh3dofconti
        """
    pass

class RawEnvVec:
    def __init__(self, batch_size: int) -> None: ...
    def at(self, arg0: int) -> RawEnv: ...
    @typing.overload
    def reset(self) -> numpy.ndarray: ...
    @typing.overload
    def reset(self, indices: numpy.ndarray) -> numpy.ndarray: ...
    @typing.overload
    def reset(self, mask: numpy.ndarray) -> numpy.ndarray: ...
    @typing.overload
    def seed(self) -> typing.List[int]: ...
    @typing.overload
    def seed(self, seed: int) -> typing.List[int]: ...
    def size(self) -> int: ...
    @typing.overload
    def step(self, action: numpy.ndarray) -> numpy.ndarray: ...
    @typing.overload
    def step(self, action: numpy.ndarray, indices: numpy.ndarray) -> numpy.ndarray: ...
    @typing.overload
    def step(self, action: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray: ...
    pass

class Veh3dofconti:
    class B_vehicle3dof_T:
        def __copy__(self) -> Veh3dofconti.B_vehicle3dof_T: ...
        def __deepcopy__(self, memo: dict) -> Veh3dofconti.B_vehicle3dof_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def Output(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def Product1(self) -> float:
            """
            :type: float
            """
        @Product1.setter
        def Product1(self, arg0: float) -> None:
            pass
        @property
        def Product1_oajp(self) -> float:
            """
            :type: float
            """
        @Product1_oajp.setter
        def Product1_oajp(self, arg0: float) -> None:
            pass
        @property
        def VectorConcatenate(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def VectorConcatenate3(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def ZeroOrderHold(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def ZeroOrderHold1(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def stateDer(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def y(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def y_fdgf(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        dtype: numpy.dtype[
            numpy.void
        ]  # value = dtype([('VectorConcatenate3', '<f8', (2,)), ('VectorConcatenate', '<f8', (4,)), ('ZeroOrderHold', '<f8', (6,)), ('Output', '<f8', (3,)), ('ZeroOrderHold1', '<f8', (3,)), ('Product1', '<f8'), ('Product1_oajp', '<f8'), ('stateDer', '<f8', (4,)), ('y', '<f8', (2,)), ('y_fdgf', '<f8', (3,))])
        pass
    class DW_vehicle3dof_T:
        def __copy__(self) -> Veh3dofconti.DW_vehicle3dof_T: ...
        def __deepcopy__(self, memo: dict) -> Veh3dofconti.DW_vehicle3dof_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def Integrator_IWORK_aike(self) -> int:
            """
            :type: int
            """
        @Integrator_IWORK_aike.setter
        def Integrator_IWORK_aike(self, arg0: int) -> None:
            pass
        @property
        def NextOutput(self) -> float:
            """
            :type: float
            """
        @NextOutput.setter
        def NextOutput(self, arg0: float) -> None:
            pass
        @property
        def RandSeed(self) -> int:
            """
            :type: int
            """
        @RandSeed.setter
        def RandSeed(self, arg0: int) -> None:
            pass
        dtype: numpy.dtype[
            numpy.void
        ]  # value = dtype({'names': ['NextOutput', 'RandSeed', 'Integrator_IWORK_aike'], 'formats': ['<f8', '<u4', '<i4'], 'offsets': [0, 8, 16], 'itemsize': 24})
        pass
    class ExtU_vehicle3dof_T:
        def __copy__(self) -> Veh3dofconti.ExtU_vehicle3dof_T: ...
        def __deepcopy__(self, memo: dict) -> Veh3dofconti.ExtU_vehicle3dof_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def Action(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def AdverAction(self) -> float:
            """
            :type: float
            """
        @AdverAction.setter
        def AdverAction(self, arg0: float) -> None:
            pass
        dtype: numpy.dtype[
            numpy.void
        ]  # value = dtype([('Action', '<f8', (3,)), ('AdverAction', '<f8')])
        pass
    class ExtY_vehicle3dof_T:
        def __copy__(self) -> Veh3dofconti.ExtY_vehicle3dof_T: ...
        def __deepcopy__(self, memo: dict) -> Veh3dofconti.ExtY_vehicle3dof_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def done(self) -> bool:
            """
            :type: bool
            """
        @done.setter
        def done(self, arg0: bool) -> None:
            pass
        @property
        def info(self) -> float:
            """
            :type: float
            """
        @info.setter
        def info(self, arg0: float) -> None:
            pass
        @property
        def obs(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def rew(self) -> float:
            """
            :type: float
            """
        @rew.setter
        def rew(self, arg0: float) -> None:
            pass
        dtype: numpy.dtype[
            numpy.void
        ]  # value = dtype({'names': ['obs', 'rew', 'done', 'info'], 'formats': [('<f8', (6,)), '<f8', '?', '<f8'], 'offsets': [0, 48, 56, 64], 'itemsize': 72})
        pass
    class InstP_vehicle3dof_T:
        def __copy__(self) -> Veh3dofconti.InstP_vehicle3dof_T: ...
        def __deepcopy__(self, memo: dict) -> Veh3dofconti.InstP_vehicle3dof_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def a_max(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def a_min(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def adva_max(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def adva_min(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def done_range(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def noise_seed(self) -> float:
            """
            :type: float
            """
        @noise_seed.setter
        def noise_seed(self, arg0: float) -> None:
            pass
        @property
        def punish_Q(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def punish_R(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def ref_A(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def ref_T(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def ref_V(self) -> float:
            """
            :type: float
            """
        @ref_V.setter
        def ref_V(self, arg0: float) -> None:
            pass
        @property
        def ref_fai(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def x_ini(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def x_max(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def x_min(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        dtype: numpy.dtype[
            numpy.void
        ]  # value = dtype([('a_max', '<f8', (3,)), ('a_min', '<f8', (3,)), ('adva_max', '<f8', (3,)), ('adva_min', '<f8', (3,)), ('done_range', '<f8', (3,)), ('punish_Q', '<f8', (4,)), ('punish_R', '<f8', (3,)), ('ref_A', '<f8', (3,)), ('ref_T', '<f8', (3,)), ('ref_fai', '<f8', (3,)), ('x_ini', '<f8', (6,)), ('x_max', '<f8', (6,)), ('x_min', '<f8', (6,)), ('noise_seed', '<f8'), ('ref_V', '<f8')])
        pass
    class X_vehicle3dof_T:
        def __copy__(self) -> Veh3dofconti.X_vehicle3dof_T: ...
        def __deepcopy__(self, memo: dict) -> Veh3dofconti.X_vehicle3dof_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def Integrator_CSTATE(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def Integrator_CSTATE_nzof(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def lateral_CSTATE(self) -> float:
            """
            :type: float
            """
        @lateral_CSTATE.setter
        def lateral_CSTATE(self, arg0: float) -> None:
            pass
        @property
        def lateral_CSTATE_iesm(self) -> float:
            """
            :type: float
            """
        @lateral_CSTATE_iesm.setter
        def lateral_CSTATE_iesm(self, arg0: float) -> None:
            pass
        dtype: numpy.dtype[
            numpy.void
        ]  # value = dtype([('Integrator_CSTATE', '<f8', (2,)), ('Integrator_CSTATE_nzof', '<f8', (4,)), ('lateral_CSTATE', '<f8'), ('lateral_CSTATE_iesm', '<f8')])
        pass
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def initialize(self) -> None: ...
    def step(self) -> None: ...
    @staticmethod
    def terminate() -> None: ...
    @property
    def vehicle3dof_B(self) -> Veh3dofconti.B_vehicle3dof_T:
        """
        :type: Veh3dofconti.B_vehicle3dof_T
        """
    @vehicle3dof_B.setter
    def vehicle3dof_B(self, arg0: Veh3dofconti.B_vehicle3dof_T) -> None:
        pass
    @property
    def vehicle3dof_DW(self) -> Veh3dofconti.DW_vehicle3dof_T:
        """
        :type: Veh3dofconti.DW_vehicle3dof_T
        """
    @vehicle3dof_DW.setter
    def vehicle3dof_DW(self, arg0: Veh3dofconti.DW_vehicle3dof_T) -> None:
        pass
    @property
    def vehicle3dof_InstP(self) -> Veh3dofconti.InstP_vehicle3dof_T:
        """
        :type: Veh3dofconti.InstP_vehicle3dof_T
        """
    @vehicle3dof_InstP.setter
    def vehicle3dof_InstP(self, arg0: Veh3dofconti.InstP_vehicle3dof_T) -> None:
        pass
    @property
    def vehicle3dof_U(self) -> Veh3dofconti.ExtU_vehicle3dof_T:
        """
        :type: Veh3dofconti.ExtU_vehicle3dof_T
        """
    @vehicle3dof_U.setter
    def vehicle3dof_U(self, arg0: Veh3dofconti.ExtU_vehicle3dof_T) -> None:
        pass
    @property
    def vehicle3dof_X(self) -> Veh3dofconti.X_vehicle3dof_T:
        """
        :type: Veh3dofconti.X_vehicle3dof_T
        """
    @vehicle3dof_X.setter
    def vehicle3dof_X(self, arg0: Veh3dofconti.X_vehicle3dof_T) -> None:
        pass
    @property
    def vehicle3dof_Y(self) -> Veh3dofconti.ExtY_vehicle3dof_T:
        """
        :type: Veh3dofconti.ExtY_vehicle3dof_T
        """
    @vehicle3dof_Y.setter
    def vehicle3dof_Y(self, arg0: Veh3dofconti.ExtY_vehicle3dof_T) -> None:
        pass
    sample_time = 0.01
    pass

__all__ = ["RawEnv", "RawEnvVec", "GymEnv", "GymEnvVec", "Veh3dofconti"]
__author__ = "hjzsj"
__version__ = "8.28"

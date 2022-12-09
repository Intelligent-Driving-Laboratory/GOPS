from __future__ import annotations
import lqs2a1
import typing
import gym.spaces.box
import numpy

__all__ = ["GymEnv", "RawEnv", "lqs2a1"]

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
    def reset(
        self,
        preinit: typing.Callable[[], None] = None,
        postinit: typing.Callable[[], None] = None,
    ) -> numpy.ndarray: ...
    @typing.overload
    def seed(self) -> typing.List[int]: ...
    @typing.overload
    def seed(self, seed: int) -> typing.List[int]: ...
    def step(self, action: numpy.ndarray) -> tuple: ...
    @property
    def model_class(self) -> lqs2a1:
        """
        :type: lqs2a1
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
    action_space: gym.spaces.box.Box  # value = Box([-5.], [5.], (1,), float64)
    metadata = {"render.modes": []}
    observation_space: gym.spaces.box.Box  # value = Box([-20. -20.], [20. 20.], (2,), float64)
    reward_range: tuple  # value = (-inf, inf)
    pass

class RawEnv:
    def __init__(self) -> None: ...
    def reset(self) -> lqs2a1.ExtY_lqs2a1_T: ...
    @typing.overload
    def seed(self) -> typing.List[int]: ...
    @typing.overload
    def seed(self, seed: int) -> typing.List[int]: ...
    def step(self, action: lqs2a1.ExtU_lqs2a1_T) -> lqs2a1.ExtY_lqs2a1_T: ...
    @property
    def model_class(self) -> lqs2a1:
        """
        :type: lqs2a1
        """
    pass

class lqs2a1:
    class B_lqs2a1_T:
        def __copy__(self) -> lqs2a1.B_lqs2a1_T: ...
        def __deepcopy__(self, memo: dict) -> lqs2a1.B_lqs2a1_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def MultiportSwitch2(self) -> float:
            """
            :type: float
            """
        @MultiportSwitch2.setter
        def MultiportSwitch2(self, arg0: float) -> None:
            pass
        @property
        def Sum1(self) -> float:
            """
            :type: float
            """
        @Sum1.setter
        def Sum1(self, arg0: float) -> None:
            pass
        dtype: numpy.dtype[
            numpy.void
        ]  # value = dtype([('MultiportSwitch2', '<f8'), ('Sum1', '<f8')])
        pass
    class ExtU_lqs2a1_T:
        def __copy__(self) -> lqs2a1.ExtU_lqs2a1_T: ...
        def __deepcopy__(self, memo: dict) -> lqs2a1.ExtU_lqs2a1_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def Action(self) -> float:
            """
            :type: float
            """
        @Action.setter
        def Action(self, arg0: float) -> None:
            pass
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
        ]  # value = dtype([('Action', '<f8'), ('AdverAction', '<f8')])
        pass
    class ExtY_lqs2a1_T:
        def __copy__(self) -> lqs2a1.ExtY_lqs2a1_T: ...
        def __deepcopy__(self, memo: dict) -> lqs2a1.ExtY_lqs2a1_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def Done(self) -> bool:
            """
            :type: bool
            """
        @Done.setter
        def Done(self, arg0: bool) -> None:
            pass
        @property
        def Reward(self) -> float:
            """
            :type: float
            """
        @Reward.setter
        def Reward(self, arg0: float) -> None:
            pass
        @property
        def State(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        dtype: numpy.dtype[
            numpy.void
        ]  # value = dtype({'names': ['State', 'Done', 'Reward'], 'formats': [('<f8', (2,)), '?', '<f8'], 'offsets': [0, 16, 24], 'itemsize': 32})
        pass
    class InstP_lqs2a1_T:
        def __copy__(self) -> lqs2a1.InstP_lqs2a1_T: ...
        def __deepcopy__(self, memo: dict) -> lqs2a1.InstP_lqs2a1_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def Q(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def R(self) -> float:
            """
            :type: float
            """
        @R.setter
        def R(self, arg0: float) -> None:
            pass
        @property
        def x_ini(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        dtype: numpy.dtype[
            numpy.void
        ]  # value = dtype([('Q', '<f8', (2,)), ('x_ini', '<f8', (2,)), ('R', '<f8')])
        pass
    class X_lqs2a1_T:
        def __copy__(self) -> lqs2a1.X_lqs2a1_T: ...
        def __deepcopy__(self, memo: dict) -> lqs2a1.X_lqs2a1_T: ...
        def __init__(self) -> None: ...
        def __repr__(self) -> str: ...
        def numpy(self) -> numpy.ndarray: ...
        @property
        def Plant_CSTATE(self) -> numpy.ndarray:
            """
            :type: numpy.ndarray
            """
        @property
        def TransferFcn_CSTATE(self) -> float:
            """
            :type: float
            """
        @TransferFcn_CSTATE.setter
        def TransferFcn_CSTATE(self, arg0: float) -> None:
            pass
        dtype: numpy.dtype[
            numpy.void
        ]  # value = dtype([('Plant_CSTATE', '<f8', (2,)), ('TransferFcn_CSTATE', '<f8')])
        pass
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def initialize(self) -> None: ...
    def skip(self, n: int) -> None:
        """
        Skip n steps, using current model input
        """
    def step(self) -> None: ...
    @staticmethod
    def terminate() -> None: ...
    @property
    def lqs2a1_B(self) -> B_lqs2a1_T:
        """
        :type: B_lqs2a1_T
        """
    @lqs2a1_B.setter
    def lqs2a1_B(self, arg0: B_lqs2a1_T) -> None:
        pass
    @property
    def lqs2a1_InstP(self) -> InstP_lqs2a1_T:
        """
        :type: InstP_lqs2a1_T
        """
    @lqs2a1_InstP.setter
    def lqs2a1_InstP(self, arg0: InstP_lqs2a1_T) -> None:
        pass
    @property
    def lqs2a1_U(self) -> ExtU_lqs2a1_T:
        """
        :type: ExtU_lqs2a1_T
        """
    @lqs2a1_U.setter
    def lqs2a1_U(self, arg0: ExtU_lqs2a1_T) -> None:
        pass
    @property
    def lqs2a1_X(self) -> X_lqs2a1_T:
        """
        :type: X_lqs2a1_T
        """
    @lqs2a1_X.setter
    def lqs2a1_X(self, arg0: X_lqs2a1_T) -> None:
        pass
    @property
    def lqs2a1_Y(self) -> ExtY_lqs2a1_T:
        """
        :type: ExtY_lqs2a1_T
        """
    @lqs2a1_Y.setter
    def lqs2a1_Y(self, arg0: ExtY_lqs2a1_T) -> None:
        pass
    sample_time = 0.05
    pass

__all__ = ("RawEnv", "GymEnv", "lqs2a1")
__author__ = "hjzsj"
__version__ = "10.16"

#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Description: Common project-wide type hints


from typing import Any, Dict

from torch import Tensor


ConfigDict = Dict[str, Any]
DataDict = Dict[str, Tensor]

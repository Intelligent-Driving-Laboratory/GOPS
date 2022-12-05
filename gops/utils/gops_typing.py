#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Common project-wide type hints
#  Update: 2021-03-10, Yuhang Zhang: create gops_typing


from typing import Any, Dict

from torch import Tensor


ConfigDict = Dict[str, Any]
DataDict = Dict[str, Tensor]
InfoDict = Dict[str, Any]

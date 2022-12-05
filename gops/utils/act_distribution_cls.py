#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Action Distribution Function
#  Update: 2021-03-05, Wenjun Zou: create action distribution function


class Action_Distribution:
    def __init__(self):
        super().__init__()

    def get_act_dist(self, logits):
        act_dist_cls = getattr(self, "action_distribution_cls")
        has_act_lim = hasattr(self, "act_high_lim")

        act_dist = act_dist_cls(logits)
        if has_act_lim:
            act_dist.act_high_lim = getattr(self, "act_high_lim")
            act_dist.act_low_lim = getattr(self, "act_low_lim")

        return act_dist

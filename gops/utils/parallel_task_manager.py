#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Task Pool Function for Ray Architecture
#  Update: 2021-03-10, Yang Guan: Create codes


import ray


class TaskPool(object):
    """
    Helper class for tracking status of many in-flight actor tasks.
    """

    def __init__(self):
        self._tasks = {}
        self._objects = {}
        self._fetching = []

    def add(self, worker, all_obj_ids):
        if isinstance(all_obj_ids, list):
            obj_id = all_obj_ids[0]
        else:
            obj_id = all_obj_ids
        self._tasks[obj_id] = worker
        self._objects[obj_id] = all_obj_ids

    def completed(self, blocking_wait=False):
        pending = list(self._tasks)
        if pending:
            ready, _ = ray.wait(pending, num_returns=len(pending), timeout=0)
            if not ready and blocking_wait:
                ready, _ = ray.wait(pending, num_returns=1, timeout=10.0)
            for obj_id in ready:
                yield self._tasks.pop(obj_id), self._objects.pop(obj_id)

    @property
    def completed_num(self):
        pending = list(self._tasks)
        if pending:
            ready, _ = ray.wait(pending, num_returns=len(pending), timeout=0)
        return len(ready)

    @property
    def count(self):
        return len(self._tasks)

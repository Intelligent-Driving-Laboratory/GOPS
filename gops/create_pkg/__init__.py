#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com


import os
import sys
import importlib

gops_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# regist algorithm
from gops.create_pkg.create_alg import register as register_alg


alg_path = os.path.join(gops_path, "algorithm")
alg_file_list = os.listdir(alg_path)

for alg_file in alg_file_list:
    if alg_file[-3:] == ".py" and alg_file[0] != "_" and alg_file != "base.py":
        alg_name = alg_file[:-3]
        mdl = importlib.import_module("gops.algorithm." + alg_name)
        register_alg(algorithm=alg_name, entry_point=getattr(mdl, alg_name.upper()))

# regist apprfunc
from gops.create_pkg.create_apprfunc import register as register_apprfunc

apprfunc_list = ["cnn", "cnn_shared", "mlp", "gauss", "poly", "rnn"]
name_list = ["DetermPolicy", "FiniteHorizonPolicy", "StochaPolicy", "ActionValue", "ActionValueDis", "StateValue"]

for apprfunc in apprfunc_list:
    for name in name_list:
        mdl = importlib.import_module("gops.apprfunc." + apprfunc)
        register_apprfunc(apprfunc=apprfunc, name=name, entry_point=getattr(mdl, name))


# regist buffer

from gops.create_pkg.create_buffer import register as register_buffer
from gops.trainer.buffer.prioritized_replay_buffer import PrioritizedReplayBuffer
from gops.trainer.buffer.replay_buffer import ReplayBuffer

register_buffer(buffer_name="PrioritizedReplayBuffer", entry_point=PrioritizedReplayBuffer)
register_buffer(buffer_name="ReplayBuffer", entry_point=ReplayBuffer)

# regist env and env model

def env_formatter(src: str, firstUpper: bool = True):
    arr = src.split("_")
    res = ""
    for i in arr:
        res = res + i[0].upper() + i[1:]

    if not firstUpper:
        res = res[0].lower() + res[1:]
    return res

from gops.create_pkg.create_env_model import register as register_env_model
from gops.create_pkg.create_env import register as register_env

# env_dir_list = ["env_gym", "env_matlab", "env_ocp", "env_pyth"]
env_dir_list = ["env_gym"]

for env_dir_name in env_dir_list:
    env_dir_abs_path = os.path.join(gops_path, "env", env_dir_name)
    file_list = os.listdir(env_dir_abs_path)
    for file in file_list:
        if file.endswith(".py") and file[0] != "_":
            try:
                env_id = file[:-3]
                mdl = importlib.import_module(f"gops.env.{env_dir_name}.{env_id}")
                env_id_camel = env_formatter(env_id)
            
                if hasattr(mdl, "env_creator"):
                    register_env(env_id=env_id, entry_point=getattr(mdl, "env_creator"))
                elif hasattr(mdl, env_id_camel):
                    register_env(env_id=env_id, entry_point=getattr(mdl, env_id_camel))
                else:
                    print(f"env {env_id} has no env_creator or {env_id_camel} in {env_dir_name}")
            except:
                RuntimeError(f"Register env {env_id} failed")

    env_dir_abs_path = os.path.join(gops_path, "env", env_dir_name, "env_model")
    file_list = os.listdir(env_dir_abs_path)
    for file in file_list:
        if file.endswith(".py") and file[0] != "_":
            env_id = file[:-3]
            mdl = importlib.import_module(f"gops.env.{env_dir_name}.env_model.{env_id}")
            env_id_camel = env_formatter(env_id)
            if hasattr(mdl, "env_model_creator"):
                register_env_model(env_id=env_id, entry_point=getattr(mdl, "env_model_creator"))
            elif hasattr(mdl, env_id_camel):
                register_env_model(env_id=env_id, entry_point=getattr(mdl, env_id_camel))
            else:
                print(f"env {env_id} has no env_model_creator or {env_id_camel} in {env_dir_name}")


# regist evaluator
from gops.create_pkg.create_evaluator import register as register_evaluator
from gops.trainer.evaluator import Evaluator
register_evaluator(evaluator_name="Evaluator", entry_point=Evaluator)

# regist sampler
from gops.create_pkg.create_sampler import register as register_sampler
from gops.trainer.sampler.off_sampler import OffSampler
from gops.trainer.sampler.on_sampler import OnSampler
register_sampler(sampler_name="OffSampler", entry_point=OffSampler)
register_sampler(sampler_name="OnSampler", entry_point=OnSampler)

# regist trainer

from gops.create_pkg.create_trainer import register as register_trainer
from gops.trainer.off_async_trainer import OffAsyncTrainer
from gops.trainer.off_sync_trainer import OffSyncTrainer
from gops.trainer.on_serial_trainer import OnSerialTrainer
from gops.trainer.on_sync_trainer import OnSyncTrainer
from gops.trainer.off_serial_trainer import OffSerialTrainer
register_trainer(trainer_name="OffAsyncTrainer", entry_point=OffAsyncTrainer)
register_trainer(trainer_name="OffSyncTrainer", entry_point=OffSyncTrainer)
register_trainer(trainer_name="OnSerialTrainer", entry_point=OnSerialTrainer)
register_trainer(trainer_name="OnSyncTrainer", entry_point=OnSyncTrainer)
register_trainer(trainer_name="OffSerialTrainer", entry_point=OffSerialTrainer)



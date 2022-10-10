from do_mpc.model import Model
from do_mpc.controller import MPC
import torch
import os
import numpy as np


class MPCController:
    def __init__(self):
        self.model = self.__init_model()
        self.optimizer = self.__init_optimizer(self.model)

    def __call__(self, init_state):
        return self.optimizer.make_step(init_state)
   
    @staticmethod
    def __init_model():
        model = Model()
        return model

    @staticmethod
    def __init_optimizer(model):
        optimizer = MPC(model)
        return optimizer


class NNController:
    def __init__(self, args, log_policy_dir):
        print(args)
        alg_name = args["algorithm"]
        alg_file_name = alg_name.lower()
        file = __import__(alg_file_name)
        ApproxContainer = getattr(file, "ApproxContainer")
        self.networks = ApproxContainer(**args)

        for filename in os.listdir(log_policy_dir + "/apprfunc"):
            if filename.endswith("_opt.pkl"):
                log_path = os.path.join(log_policy_dir, "apprfunc", filename)
        self.networks.load_state_dict(torch.load(log_path))

    def __call__(self, obs):
        batch_obs = torch.from_numpy(np.expand_dims(obs, axis=0).astype("float32"))
        logits = self.networks.policy(batch_obs)
        action_distribution = self.networks.create_action_distributions(logits)
        action = action_distribution.mode()
        action = action.detach().numpy()[0]
        return action
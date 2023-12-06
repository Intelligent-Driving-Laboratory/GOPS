from gops.env.env_gen_ocp.quadrotor_1dof_tracking_stablization import Quadrotor1dofTrackingStablization
        
def env_creator(**kwargs):
    return Quadrotor1dofTrackingStablization(task = "TRAJ_TRACKING", **kwargs)



    



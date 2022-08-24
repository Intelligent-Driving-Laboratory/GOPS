import os

MUJOCO = [
    "gym_ant",
    "gym_halfcheetah",
    "gym_hopper",
    "gym_humanoid",
    "gym_humanoidstandup",
    "gym_inverteddoublependulum",
    "gym_invertedpendulum",
    "gym_pusher",
    "gym_reacher",
    "gym_swimmer",
    "gym_walker2d",
]

CLASSIC = [
    "gym_cartpoleconti",
    "gym_mountaincarconti",
    "gym_acrobot",
    "gym_cartpole",
    "gym_mountaincar",
    "gym_pendulum",
]

TOY_TEXT = [
    "gym_blackjack",
    "gym_frozenlake88",
    "gym_frozenlake",
    "gym_taxi",
]

BOX2D = [
    "gym_bipedalwalkerhardcore",
    "gym_carracing",
    "gym_lunarlander",
    "gym_bipedalwalker",
    "gym_lunarlanderconti",
]

ATARI = [
    "gym_breakout",
    "gym_boxing",
    "gym_enduro",
    "gym_phoenix",
    "gym_spaceinvaders",
]

SIMU = [
    "simu_aircraftconti",
    "simu_veh3dofconti",
    "simu_cartpoleconti",
    "simu_doublemassconti",
]

PYTH = [
    "pyth_acc",
    "pyth_carfollowing",
    "pyth_intersection",
    "pyth_inverteddoublependulum",
    "pyth_linearquadratic",
    "pyth_mobilerobot",
    "pyth_veh2dofconti",
    "pyth_veh3dofconti",
    "pyth_trackingcar",
    "pyth_pcc_caramt",
    "pyth_pcc_carcvt",
    "pyth_pcc_truckamt",
    "pyth_pcc_trucklcf",
]



def get_env_model_files():
    env_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_list = []
    model_list = []
    for f in os.listdir(env_folder_path):
        file_path = os.path.join(env_folder_path, f)
        if os.path.isdir(file_path):
            pass
        elif os.path.splitext(file_path)[1] == ".py" and f.startswith(("gym", "pyth", "simu")):
            if f.endswith("data.py"):
                env_list.append(os.path.splitext(f)[0])
            elif f.endswith("model.py"):
                model_list.append(os.path.splitext(f)[0])

    return env_list, model_list


if __name__ == "__main__":
    em = get_env_model_files()
    print(em)

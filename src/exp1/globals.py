import sys
import numpy as np
import torch

from scipy.spatial.transform import Rotation as R


def add_paths():
    sys.path.insert(0, "/home/ruic/Documents/opa")
    sys.path.insert(0, "/home/ruic/Documents/opa/opa_comparison/src")
    sys.path.insert(0, "/Users/Alvin/Documents/opa")
    sys.path.insert(0, "/Users/Alvin/Documents/opa/opa_comparison/src")


add_paths()

# from exp_utils import BOX_ID, CAN_ID, BOX_MASS, CAN_MASS


num_objects = 1

ROOT_PATH = "/home/ruic/Documents/opa/opa_comparison/src/exp1"
HOME_POSE = np.array([0.373, 0.1, 0.13, 0.707, 0.707, 0, 0])
HOME_JOINTS = np.array([-0.705, 0.952, -1.663, -1.927, 2.131, 1.252, -0.438])
POS_DIM = 3
DROP_OFF_OFFSET = np.array([0.0, 0.0, -0.1])
goal_radius_scale_custom = 0.6

T = 20.0  # taken from FERL yaml file

# Item ID's (if item == can, slide into grasp pose horizontally)
BOX_ID = 0
CAN_ID = 1

# Item masses
BOX_MASS = 0.3
CAN_MASS = 1.0

item_ids = [
    BOX_ID,
    BOX_ID,
    BOX_ID,
]

extra_masses = [
    BOX_MASS,
    BOX_MASS,
    BOX_MASS,
]

# Item pickup poses
start_poses = [
    # 1st: human perturbation
    np.array([0.2, 0.35, -0.08]),
    # (demo 1)
    np.array([0.674, -0.138, 0.067]),
    # (demo 2)
    np.array([0.2, 0.35, -0.08]),
]
start_ori_quats = [
    # 1st: human perturbation
    np.array([0, 1., 0, 0]),
    # (demo 1)
    np.array([0, 1., 0, 0]),
    # (demo 2)
    np.array([0, 1., 0, 0]),
]
start_joints_all = [
    # 1st: human perturbation
    np.array([270.271, 68.32, 221.0, 241.136, 91.637, 36.716, 18.348]),
    # (demo 1)
    np.array([344.215, 52.676, 213, 264.5, 129.6, 26.3, 67.4]),
    # (sim 1)
    np.array([270.271, 68.32, 221.0, 241.136, 91.637, 36.716, 18.348]),
]

# Item dropoff poses
goal_poses = [
    # 1st: human perturbation
    np.array([0.4, -0.475, 0.1]),
    # (demo 1)
    # np.array([-0.308, -0.43, .037]),
    np.array([-0.368, -0.484, -0.05]),
    # (demo 2)
    np.array([0.4, -0.475, 0.1]),
]
goal_ori_quats = start_ori_quats
goal_ori_quats[1] = R.from_euler("XYZ", [128.8, -4.98, 54.3], degrees=True).as_quat()
goal_joints_all = [
    # 1st: human perturbation
    np.array([20, 67.329, 249.304, 289.5, 117.57, 80.329, 92.4]),
    # (demo 1)
    np.array([170.49, 46.7, 185.3, 262.1, 164.6, 17.58, 102.1]),
    # (sim 1)
    np.array([20, 67.329, 249.304, 289.5, 117.57, 80.329, 92.4]),
]

inspection_poses = [
    # 1st: human perturbation
    np.array([0.6, 0.1, 0.05]),  # sitting at center of table
    # (demo 1)
    np.array([0.381, -0.834, 0.5]),  # sitting at LHS of table
    # np.array([-0.089, -0.724, 0.022]),  # sitting at LHS of table
    # (demo 2)
    np.array([0.6, 0.1, 0.5]),  # standing at center of table

    np.array([0.6, -0.2, 0.4]),  
    np.array([0.6, 0.2, -0.1]),   # standing at another corner of table
]

setting2_offset = R.from_euler("XYZ", [0, 0, -90], degrees=True)

inspection_ori_quats = [
    # 1st: human perturbation
    R.from_euler("zyx", [0, 30, 0], degrees=True).as_quat(),
    # (demo 1)
    (setting2_offset * R.from_euler("zyx", [0, 30, 0], degrees=True)).as_quat(),
    # (demo 2)
    R.from_euler("zyx", [0, 30, 0], degrees=True).as_quat(),
]

# start_poses = start_poses[1:]
# start_ori_quats = start_ori_quats[1:]
# start_joints_all = start_joints_all[1:]
# goal_poses = goal_poses[1:]
# goal_joints_all = goal_joints_all[1:]
# goal_ori_quats = goal_ori_quats[1:]
# inspection_poses = inspection_poses[1:]
# inspection_ori_quats = inspection_ori_quats[1:]

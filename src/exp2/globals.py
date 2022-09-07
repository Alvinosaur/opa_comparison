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
    BOX_ID,
]

extra_masses = [
    BOX_MASS,
    BOX_MASS,
    BOX_MASS,
    BOX_MASS
]

"""I think live, moving obstacles may not be necessary, may just be overkill
because the other methods should still fail to generalize to different
object positions, even if not major change"""

# Item pickup poses
start_poses = [
    # perturbation obstacle 1
    np.array([0.2, 0.35, -0.08]),
    np.array([0.2, 0.465, -0.08]),

    # perturbation obstacle 1+2
    np.array([0.2, 0.465, -0.08]),
    np.array([0.2, 0.465, -0.08]),

    # diff start/goal and obstacle
    #    TODO:
]
start_ori_quats = [
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),
]
start_joints_all = [
    np.array([270.271, 68.32, 221.0, 241.136, 91.637, 36.716, 18.348]),
    np.array([270.271, 68.32, 221.0, 241.136, 91.637, 36.716, 18.348]),
    np.array([270.271, 68.32, 221.0, 241.136, 91.637, 36.716, 18.348]),
    np.array([270.271, 68.32, 221.0, 241.136, 91.637, 36.716, 18.348]),
]

# Item dropoff poses
goal_poses = [
    np.array([0.4, -0.475, 0.1]),
    np.array([0.4, -0.475, 0.1]),
    np.array([0.4, -0.475, 0.1]),
    np.array([0.4, -0.475, 0.1]),

]
goal_ori_quats = start_ori_quats
goal_joints_all = [
    np.array([20, 67.329, 249.304, 289.5, 117.57, 80.329, 92.4]),
    np.array([20, 67.329, 249.304, 289.5, 117.57, 80.329, 92.4]),
    np.array([20, 67.329, 249.304, 289.5, 117.57, 80.329, 92.4]),
    np.array([20, 67.329, 249.304, 289.5, 117.57, 80.329, 92.4]),

    # TODO:

]

obstacle_poses = [
    # # obstacle 1
    # np.array([0.34, -0.01, 0.1]),
    # np.array([0.34, -0.01, 0.1]),

    # obstacle 2
    np.array([0.36, -0.026, 0.37]),
    np.array([0.36, -0.026, 0.37]),

]

obstacle_ori_quats = [np.array([0, 0, 0, 1.])
                      for _ in range(len(obstacle_poses))]


goal_radius_scale_custom = 0.4

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
goal_radius_scale_custom = 0.5

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
    np.array([0.2, 0.35, -0.08]),
    np.array([0.2, 0.35, -0.08]),

    # extreme diff settings
    np.array([0.6, -0.4, 0.1]),
    np.array([0.6, -0.4, 0.1]),
    np.array([-0.6, 0.2, 0.2]),

]
start_ori_quats = [
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),

    # some random quats
    np.array([0.46765938, 0.26540881, -0.84267767, 0.0273354]),
    np.array([-0.91210267, 0.22798313, -0.0389284, -0.33849224]),
    np.array([0.54350796, -0.02228003, -0.83859095, -0.02946044]),
]
start_joints_all = [
    np.array([270.271, 68.32, 221.0, 241.136, 91.637, 36.716, 18.348]),
    np.array([270.271, 68.32, 221.0, 241.136, 91.637, 36.716, 18.348]),
    np.array([270.271, 68.32, 221.0, 241.136, 91.637, 36.716, 18.348]),

    np.array([348.12529893, 104.65223782, 338.29911603,
             45.98672921, 35.68942927, 121.19936588, 161.72245029]),
    np.array([3.53011034, 29.38366487, 3.71409297, 106.76884702,
             40.01603115, 55.50209634, 130.19839]),
    np.array([242.95258532, 110.01893027, 83.050402, 261.09189552,
              300.88951033, 19.43122191, 21.36529366])
]

# Item dropoff poses
goal_poses = [
    np.array([0.4, -0.475, 0.1]),
    np.array([0.4, -0.475, 0.1]),
    np.array([0.4, -0.475, 0.1]),

    np.array([0.1, 0.0, 0.7]),
    np.array([0.1, 0.0, 0.7]),
    np.array([0.3, 0.0, -0.4]),
]
goal_ori_quats = start_ori_quats
goal_joints_all = [
    np.array([20, 67.329, 249.304, 289.5, 117.57, 80.329, 92.4]),
    np.array([20, 67.329, 249.304, 289.5, 117.57, 80.329, 92.4]),
    np.array([20, 67.329, 249.304, 289.5, 117.57, 80.329, 92.4]),

    np.array([3.49899313e+02, 4.95570056e+01, 3.58645345e+02, 1.27871874e-02,
              9.90871568e+00, 2.33767408e+02, 1.71073631e+02]),
    np.array([22.10433667, 329.03517559, 325.08487396, 79.76884971, 43.88343671,
              97.38938634, 174.01621357]),
    np.array([2.71487593, 125.39812037, 56.24615154, 42.09632447,
             271.53675886, 112.47003623, 146.30995758])
]

inspection_poses = [
    np.array([0.6, 0.1, 0.05]),  # sitting at center of table
    np.array([0.6, -0.2, 0.4]),   # standing at one corner of table
    np.array([0.6, 0.2, -0.1]),   # standing at another corner of table

    # extreme diff settings
    np.array([0.5, -0.1, 0.4]),
    np.array([0.0, 0.0, 0.1]),
    np.array([0.2, 0.15, -0.1]),
]
inspection_ori_quats = [
    R.from_euler("zyx", [0, 30, 0], degrees=True).as_quat(),
    R.from_euler("zyx", [-20, -20, 0], degrees=True).as_quat(),
    R.from_euler("zyx", [20, 40, 0], degrees=True).as_quat(),

    # some random quats
    np.array([-0.08406316, -0.49222364, -0.60447908, 0.62068858]),
    np.array([0.31430839, -0.61724114, 0.12140935, 0.71097354]),
    np.array([-0.14625042, -0.56405624, -0.60235748, -0.5455427]),
]

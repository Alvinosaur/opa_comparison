"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np
import argparse
import os
from scipy.spatial.transform import Rotation as R
import json
import re
from tqdm import tqdm

import torch
import sys

from globals import *
add_paths()  # sets the paths for the below imports

from exp_utils import *

from unified_trajopt import TrajOptBase
from trajectory import Trajectory
from ferl_network import DNN

import signal

signal.signal(signal.SIGINT, sigint_handler)

World2Net = 10.0
Net2World = 1 / World2Net

DEBUG = True
dstep = 0.05
ros_delay = 0.1

DEVICE = "cuda:0"
print("DEVICE: %s" % DEVICE)

# Hyperparameters
T = 3.0
TRAJ_LEN = 30  # fixed number of wayponts for trajopt
waypts_time = np.linspace(0, T, TRAJ_LEN)
adapt_num_epochs = 5


class TrajOptExp(TrajOptBase):
    # experiment-specific oracle reward function used to generate "optimal"
    # human behavior
    POS_WEIGHT = 1.0
    ROT_WEIGHT = 0.75

    def __init__(self, human_pose_euler, context_dim, *args, **kwargs):
        super(TrajOptExp, self).__init__(*args, **kwargs)
        self.human_pose_euler = human_pose_euler

        # rotate relative to human facing exactly opposite of human
        # so desired orientation is 180 degrees rotated aboutu human's z-axis.
        self.desired_present_item_rot = (R.from_euler("xyz", human_pose_euler[3:7]) *
                                         R.from_euler("xyz", [0, 0, 180], degrees=True)).as_quat()

    # custom process context function
    def process_context(self, robot, context):
        human_pose_quat = context
        if not self.use_state_features:
            return human_pose_quat
        else:
            dist = np.linalg.norm(robot[0:3] - human_pose_quat[0:3])
            vec = (human_pose_quat[0:3] - robot[0:3]) / dist
            return np.concatenate([dist, vec, human_pose_quat[3:7]])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store',
                        type=str, help="trained model name", default="policy_3D")
    parser.add_argument('--loaded_epoch', action='store',
                        type=int, default=100)
    parser.add_argument('--view_ros', action='store_true',
                        help="visualize 3D scene with ROS Rviz")
    parser.add_argument('--exp_folder', action='store', type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    ########################################################
    args = parse_arguments()
    argparse_dict = vars(args)
    ROOT_LOAD_FOLDER = "/home/ruic/Documents/opa/opa_comparison/src/exp2/"
    save_folder = os.path.join(ROOT_LOAD_FOLDER, args.exp_folder)
    # os.makedirs(save_folder, exist_ok=True)

    # Define reward model
    state_dim = 3 + 3  # (pos, rot euler)
    # context: human pose
    context_dim = 3 + 3
    input_dim = state_dim + context_dim  # robot pose, human pose
    ferl_dnn = DNN(nb_layers=3, nb_units=128, input_dim=12)
    ferl_dnn.load_state_dict(torch.load(os.path.join(ROOT_LOAD_FOLDER, args.exp_folder, "model_0.pth")))
    ferl_dnn.to(DEVICE)

    it = 0
    pose_error_tol = 0.1
    max_pose_error_tol = 0.2  # too far to be considered converged and not moving
    del_pose_tol = 0.005  # over del_pose_interval iterations
    num_exps = len(start_poses)
    num_rand_trials = 10
    pbar = tqdm(total=num_exps * num_rand_trials)
    for exp_iter in range(num_exps):
        # set extra mass of object to pick up
        # exp_iter = num_exps - 1
        # exp_iter = min(exp_iter, num_exps - 1)
        # extra_mass = extra_masses[exp_iter]

        # Set start robot pose
        start_pos = start_poses[exp_iter]
        start_ori_quat = start_ori_quats[exp_iter]
        start_ori_euler = R.from_quat(start_ori_quat).as_euler("XYZ")
        start_pose = np.concatenate([start_pos, start_ori_euler])
        start_pose_quat = np.concatenate([start_pos, start_ori_quat])

        # Set goal robot pose
        goal_pos = goal_poses[exp_iter]
        goal_ori_quat = goal_ori_quats[exp_iter]
        goal_ori_euler = R.from_quat(goal_ori_quat).as_euler("XYZ")
        goal_pose = np.concatenate([goal_pos, goal_ori_euler])
        goal_pose_quat = np.concatenate([goal_pos, goal_ori_quat])

        # Set inspection pose
        obstacle_pos = obstacle_poses[exp_iter]
        obstacle_ori_quat = obstacle_ori_quats[exp_iter]
        obstacle_ori_euler = R.from_quat(obstacle_ori_quat).as_euler("XYZ")
        obstacle_pose_euler = np.concatenate(
            [obstacle_pos, obstacle_ori_euler])

        for rand_trial in range(num_rand_trials):
            # add small random noise to start/goal/objects
            def rand_pos_noise():
                return np.random.normal(loc=0, scale=0.05, size=3)
            def rand_rot_euler_noise():
                return np.random.normal(loc=0, scale=5 * np.pi / 180, size=3)
            start_pose_noisy = np.concatenate([
                start_pose[0:3] + rand_pos_noise(),
                start_pose[3:] + rand_rot_euler_noise()
            ])
            goal_pose_noisy = np.concatenate([
                goal_pose[0:3] + rand_pos_noise(),
                goal_pose[3:] + rand_rot_euler_noise()
            ])
            obstacle_pose_noisy = np.concatenate([
                obstacle_pose_euler[0:3] + rand_pos_noise(),
                obstacle_pose_euler[3:] + rand_rot_euler_noise()
            ])
            
            trajopt = TrajOptExp(home=start_pose_noisy,
                                goal=goal_pose_noisy,
                                human_pose_euler=obstacle_pose_noisy,
                                context_dim=context_dim,
                                use_state_features=False,
                                waypoints=TRAJ_LEN)
            traj = Trajectory(
                waypts=trajopt.optimize(
                    context=obstacle_pose_noisy, reward_model=ferl_dnn),
                waypts_time=waypts_time)

            np.savez(os.path.join(save_folder, f"ee_pose_traj_iter_{exp_iter}_rand_trial_{rand_trial}.npz"), traj=traj.waypts, start_pose=start_pose_noisy, goal_pose=goal_pose_noisy, obstacle_pose=obstacle_pose_noisy)

            pbar.update(1)

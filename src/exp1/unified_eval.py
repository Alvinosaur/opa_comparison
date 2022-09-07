"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np
import os
import argparse
import json
import re

import torch
import sys

from globals import *
add_paths()  # sets the paths for the below imports

from exp_utils import *

from unified_trajopt import TrajOptBase
from unified_reward import TrainReward
from trajectory import Trajectory

import signal

signal.signal(signal.SIGINT, sigint_handler)

World2Net = 10.0
Net2World = 1 / World2Net

DEBUG = True
dstep = 0.05
ros_delay = 0.1

DEVICE = "cpu"
print("DEVICE: %s" % DEVICE)

# Hyperparameters
TRAJ_LEN = 10  # fixed number of wayponts for trajopt
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
    parser.add_argument('--collected_folder', action='store', type=str)
    parser.add_argument('--use_state_features', action='store_true',
                        help="context are the same state features that OPA uses, not raw object state")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    ########################################################
    args = parse_arguments()
    argparse_dict = vars(args)

    # define save path
    save_folder = f"exp1/unified_saved_trials_inspection/eval"
    os.makedirs(save_folder, exist_ok=True)

    # Define reward model
    state_dim = 3 + 3  # (pos, rot euler)
    # context: human pose
    context_dim = 1 + 3 + 3 if args.use_state_features else 3 + 3
    input_dim = state_dim + context_dim  # robot pose, human pose
    rm1 = TrainReward(model_dim=(input_dim, 128),
                      epoch=500, traj_len=TRAJ_LEN, device=DEVICE)

    it = 0
    pose_error_tol = 0.1
    max_pose_error_tol = 0.2  # too far to be considered converged and not moving
    del_pose_tol = 0.005  # over del_pose_interval iterations
    num_exps = len(start_poses)
    # num_exps = 3
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
        inspection_pos = inspection_poses[exp_iter]
        inspection_ori_quat = inspection_ori_quats[exp_iter]
        inspection_ori_euler = R.from_quat(inspection_ori_quat).as_euler("XYZ")
        inspection_pose_euler = np.concatenate(
            [inspection_pos, inspection_ori_euler])

        trajopt = TrajOptExp(home=start_pose,
                             goal=goal_pose,
                             human_pose_euler=inspection_pose_euler,
                             context_dim=context_dim,
                             use_state_features=args.use_state_features,
                             waypoints=TRAJ_LEN)
        traj = Trajectory(
            waypts=trajopt.optimize(
                context=inspection_pose_euler, reward_model=rm1),
            waypts_time=waypts_time)
        local_target_pos = traj.waypts[0, 0:3]
        local_target_ori_quat = R.from_euler(
            "XYZ", traj.waypts[0, 3:]).as_quat()

        # initialize target pose variables
        cur_pos = np.copy(start_pose[0:3])
        cur_ori_euler = np.copy(start_pose[3:])
        cur_ori_quat = R.from_euler("XYZ", cur_ori_euler).as_quat()

        intervene_count = 0
        pose_error = 1e10
        del_pose = 1e10
        del_pose_running_avg = RunningAverage(length=5, init_vals=1e10)

        ee_pose_traj = []
        prev_pose_quat = None
        step = 0
        max_steps = 100
        dt = 0.3
        while (pose_error > pose_error_tol and
                (pose_error > max_pose_error_tol) and step < max_steps):
            step += 1
            # calculate next action to take based on planned traj
            cur_pose = np.concatenate([cur_pos, cur_ori_euler])
            cur_pose_quat = np.concatenate([cur_pos, cur_ori_quat])
            pose_error = calc_pose_error(
                goal_pose_quat, cur_pose_quat, rot_scale=0)

            ee_pose_traj.append(cur_pose_quat.copy())

            local_target_pose = traj.interpolate(
                t=step * dt).flatten()
            local_target_pos = local_target_pose[0:3]
            local_target_ori_quat = R.from_euler(
                "XYZ", local_target_pose[3:]).as_quat()

            cur_pos = local_target_pos
            cur_ori_quat = local_target_ori_quat
            cur_ori_euler = local_target_pose[3:]

            print("dist_to_goal: ", pose_error)
            prev_pose_quat = np.copy(cur_pose_quat)

        # Save robot traj and intervene traj
        np.save(f"{save_folder}/ee_pose_traj_iter_{exp_iter}.npy", ee_pose_traj)

        print(
            f"Finished! Error {pose_error} vs tol {pose_error_tol}, \nderror {del_pose_running_avg.avg} vs tol {del_pose_tol}")
        print("Opening gripper to release item")

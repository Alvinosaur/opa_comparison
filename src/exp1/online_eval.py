		
"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from math import degrees
from operator import is_
import numpy as np
import os
import argparse
import json
import re
from tqdm import tqdm
import time

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

T = 3.0
DEBUG = True
dstep = 0.05
ros_delay = 0.1

DEVICE = "cpu"
print("DEVICE: %s" % DEVICE)

# Hyperparameters
TRAJ_LEN = 10  # fixed number of wayponts for trajopt
waypts_time = np.linspace(0, T, TRAJ_LEN)
adapt_num_epochs = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_adaptation_time_sec',
                        action='store', type=float)
    parser.add_argument('--num_perturbs', action='store',
                        type=int, required=True)
    parser.add_argument('--is_expert', action='store_true',
                        help="use is_expert, true features")
    args = parser.parse_args()

    return args

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

    # trajectory cost function
    def trajcost(self, reward_model, context, xi):
        states = xi.reshape(self.n_waypoints, self.state_dim)
        states = np.vstack(states)
        R_learned = reward_model.reward(states)

        avoid_stuck_weight = 0  # TODO: tune
        R_avoid_stuck = avoid_stuck_weight * np.linalg.norm(
            states[1:, 0:3] - states[0:-1, 0:3], axis=-1).sum()
        cost = -(R_learned + R_avoid_stuck)
        self.count += 1
        if self.count > self.max_iter:
            return 0
        return cost


class PredefinedReward(object):
    def __init__(self, is_expert):
        # some poses to either avoid or attract to
        self.positions = []
        self.orientations = []  # quaternions
        self.pos_weights = np.array([])  # to be optimized
        self.ori_weights = np.array([])  # to be optimized
        self.human_pose = None
        self.desired_rot_offset = None   # only during expert mode
        self.alpha = 0.001
        self.is_expert = is_expert
        self.iter = 0
        """
        Expert: 
        - only have the human position
        - only have the exact desired rotation (calculated given human orientation and desired rotational offset)

        Non-Expert:
        - have positions of human and other objects (5) in the scene
        - given orientation of human and those other objects with discretized rotational offsets (exact offset is unknown, try to discretely cover the entire space of possible offsets, 30 degrees)
        
        Fo

        """
        
        self.rot_offsets = []
        # for rx in range(0, 361, 30):
        #     for ry in range(0, 361, 30):
        #         for rz in range(0, 361, 30):
        #             rot_offset = R.from_euler("XYZ", [rx, ry, rz], degrees=True)
        #             self.rot_offsets.append(rot_offset)
        for _ in range(30):
            self.rot_offsets.append(rand_quat())

        if self.is_expert: 
            self.num_random_pos = 0
            self.num_random_rot = 0
        else:
            self.num_random_pos = 5
            self.num_random_rot = 5
            self.positions = [np.random.normal(loc=0, scale=0.5, size=3)
                                for _ in range(self.num_random_pos)]
            self.pos_weights = np.ones(self.num_random_pos)  # -dist

            self.orientations = [rand_quat()
                                for _ in range(self.num_random_rot)]
            self.ori_weights = np.ones(len(self.rot_offsets) * len(self.orientations))
        

    def dist(self, traj_euler, pos):
        return np.linalg.norm(traj_euler[:, 0:3] - pos, axis=-1).sum()

    def ori_dist(self, traj_euler, ori_quat):
        traj_quat = R.from_euler("XYZ", traj_euler[:, 3:]).as_quat()
        if self.is_expert:
            # calculate desired rot give human ori andd ddesired offset
            desired_rot = (R.from_quat(ori_quat) *
                   R.from_quat(self.desired_rot_offset)).as_quat()

            return [np.arccos(np.clip(np.abs(traj_quat @ desired_rot), 0, 1)).mean(), ]
        else:
            possible_rots = [(R.from_quat(ori_quat) *
                   R.from_quat(rot_offset)).as_quat()
                for rot_offset in self.rot_offsets]
            return [np.arccos(np.clip(np.abs(traj_quat @ rot), 0, 1)).mean() for rot in possible_rots]
            
    def reward(self, x, ret_reward=True):
        assert self.human_pose is not None
        traj = x
        pos_dists = np.array([self.dist(traj, pos) for pos in self.positions])
        # neg_pos_dists = -pos_dists

        # NOTE: no concept of avoid/attract for ori, only try minimize ori dist or not, so only apply -1
        ori_dists = np.concatenate([self.ori_dist(traj, ori) for ori in self.orientations])

        if ret_reward:
            # if self.iter % 50 == 0:
            #     print(self.iter, self.pos_weights @ pos_dists, rot_weight * self.ori_weights @
            # return reward = -1*cost
            res = -1 * (self.pos_weights @ np.concatenate([pos_dists]) + self.ori_weights @ ori_dists)
            return res
        else:
            # formualted as cost minimizattion so postivie
            return +1 * np.concatenate([pos_dists, ori_dists])

    def set_human_pose(self, human_pose):
        self.human_pose = human_pose

        if len(self.positions) == self.num_random_pos:
            self.positions.append(human_pose[0:3])
            # add new pos weight for dist to human
            self.pos_weights = np.append(self.pos_weights, 1)
        else:
            self.positions[-1] = human_pose[0:3]

        if len(self.orientations) == self.num_random_rot:
            self.orientations.append(human_pose[3:])
            if self.is_expert:
                self.ori_weights = np.append(self.ori_weights, 1)
            else:
                self.ori_weights = np.append(self.ori_weights, [1] * len(self.rot_offsets))
        else:
            self.orientations[-1] = human_pose[3:]

    def set_desired_rot_offset(self, desired_rot_offset):
        assert self.is_expert
        self.desired_rot_offset = desired_rot_offset

    def generate_orig(self, expert_traj, goal_pose_euler):
        # Generate the original trajectories starting from the perturbation pose
        # that will be used to compare with the perturbation trajectories
        initial_pose = expert_traj[0]
        trajopt = TrajOptExp(home=initial_pose,
                             goal=goal_pose_euler,
                             human_pose_euler=expert_traj[-1],
                             context_dim=context_dim,
                             use_state_features=False,
                             waypoints=TRAJ_LEN, max_iter=100)
        
        orig_traj = Trajectory(waypts=trajopt.optimize(
                context=None, reward_model=self),
            waypts_time=waypts_time).waypts

        return orig_traj

    def update_weights_one_step(self, expert, goal_pose_euler, method):
        orig = self.generate_orig(expert, goal_pose_euler)
        orig_feats = self.reward(orig, ret_reward=False)
        expert_feats = self.reward(expert, ret_reward=False)
        update = expert_feats - orig_feats
        # print(expert_feats, orig_feats)
        # import ipdb
        # ipdb.set_trace()

        # print(expert_feats)
        # print(orig_feats)
        # print(np.array2string(update[:6], precision=2))
        # print()
        if method == "max":
            max_pos_idx = np.argmax(np.fabs(update[0:len(self.pos_weights)]))
            max_ori_idx = np.argmax(np.fabs(update[len(self.ori_weights):]))
            
            self.pos_weights[max_pos_idx] -= self.alpha * update[max_pos_idx]
            self.ori_weights[max_ori_idx] -= self.alpha * update[max_ori_idx]
        else:
            self.pos_weights -= self.alpha * update[0:len(self.pos_weights)]
            self.ori_weights -= self.alpha * update[len(self.pos_weights):]

        # ensure non-negative weights otherwise some non-important 
        # weights could be set to negative and optimized incorrectly
        # any non-important features should have 0 weight
        # and only important features have high positive weight
        # print()
        # print(self.pos_weights)
        # print(self.ori_weights)
        
        # print(np.array2string(update[0:len(self.pos_weights)], precision=2))
        # import ipdb
        # ipdb.set_trace()
        # print(self.pos_weights)
       
        # print(self.ori_weights)
        # print()
        # self.pos_weights -= np.min(self.pos_weights)
        # if len(self.ori_weights) > 1:
        #     self.ori_weights -= np.min(self.ori_weights)
        # self.pos_weights = np.clip(self.pos_weights, 0, np.Inf)
    #     if not self.is_expert:
    #         self.pos_weights = np.exp(np.array([-35.59819445, -61.38818335, -24.51782259, -35.46247483,
    #    -54.03993968, -10.35149632]))
        # print(self.ori_weights)
        self.ori_weights = np.clip(self.ori_weights, 0, np.Inf)
        self.pos_weights = np.clip(self.pos_weights, 0, np.Inf)

        
# class OracleReward(PredefinedReward):
#     def __init__(self):
#         super().__init__()

#     def set_desired_pose(self, human_pos, desired_rot_quat):
#         self.positions = [human_pos]
#         self.orientations = [desired_rot_quat]
#         self.pos_weights = np.ones(1)
#         self.ori_weights = np.ones(1)

def rand_quat() -> np.ndarray:
    """
    Generate a random quaternion: http://planning.cs.uiuc.edu/node198.html
    :return: quaternion np.ndarray(4,)
    """
    u, v, w = np.random.uniform(0, 1, 3)
    return np.array([np.sqrt(1 - u) * np.sin(2 * np.pi * v),
                     np.sqrt(1 - u) * np.cos(2 * np.pi * v),
                     np.sqrt(u) * np.sin(2 * np.pi * w),
                     np.sqrt(u) * np.cos(2 * np.pi * w)])


def run_adaptation(rm: PredefinedReward, desired_rot_offset, save_folder, collected_folder, num_perturbs, max_adaptation_time_sec):
    files = os.listdir(collected_folder)
    exp_iter = None
    all_perturb_pose_traj = []
    if num_perturbs == 1:
        perturb_pose_traj_quat = np.load(os.path.join(
                        collected_folder, "perturb_traj_iter_0_num_0.npy"))
        perturb_pose_traj_euler = np.hstack([
                        perturb_pose_traj_quat[:, 0:3],
                        R.from_quat(perturb_pose_traj_quat[:, 3:]).as_euler("XYZ")
                    ])

        all_perturb_pose_traj.append(perturb_pose_traj_euler)
    else:
        for f in files:
            matches = re.findall("perturb_traj_iter_(\d+)_num_\d+.npy", f)
            if len(matches) > 0:

                if len(all_perturb_pose_traj) < num_perturbs:
                    perturb_pose_traj_quat = np.load(os.path.join(
                        collected_folder, f))

                    perturb_pose_traj_euler = np.hstack([
                        perturb_pose_traj_quat[:, 0:3],
                        R.from_quat(perturb_pose_traj_quat[:, 3:]).as_euler("XYZ")
                    ])

                    all_perturb_pose_traj.append(perturb_pose_traj_euler)
    num_perturbs = len(all_perturb_pose_traj)

    exp_iter = 0  # always perturbation at 0th iter setting
    # Set goal pose
    goal_pos = goal_poses[exp_iter]
    goal_ori_quat = goal_ori_quats[exp_iter]
    goal_pose_euler = np.concatenate(
        [goal_pos, R.from_quat(goal_ori_quat).as_euler("XYZ")])

    # Set inspection pose
    inspection_pos = inspection_poses[exp_iter]
    inspection_ori_quat = inspection_ori_quats[exp_iter]
    inspection_ori_euler = R.from_quat(inspection_ori_quat).as_euler("XYZ")
    inspection_pose_euler = np.concatenate(
        [inspection_pos, inspection_ori_euler])
    inspection_pose_quat = np.concatenate(
        [inspection_pos, inspection_ori_quat])

    start_time = time.time()

    # Data processing
    all_expert_pose_traj = []
    all_orig_predicted_pose_traj = []
    num_wpts = 40
    start_time = time.time()

    for perturb_pose_traj_euler in all_perturb_pose_traj:
        waypts_time=np.linspace(0, num_wpts, len(perturb_pose_traj_euler))
        perturb_pose_traj_euler = Trajectory(
                waypts=perturb_pose_traj_euler,
                waypts_time=waypts_time
            ).downsample(num_waypts=num_wpts).waypts
        all_expert_pose_traj.append(perturb_pose_traj_euler)
    all_expert_pose_traj = np.array(all_expert_pose_traj)

    # This took 97 sec... not sure if should include 97/N or not
    # print("time spent: ", time.time() - start_time)
    # np.save("exp1/all_orig_predicted_pose_traj.npy", all_orig_predicted_pose_traj)
    # np.save("exp1/all_expert_pose_traj.npy", all_expert_pose_traj)
    # # exit()
    # all_orig_predicted_pose_traj = np.load("exp1/online_saved_trials_inspection/all_orig_predicted_pose_traj.npy")
    # all_expert_pose_traj = np.load("exp1/online_saved_trials_inspection/all_expert_pose_traj.npy")

    rm.set_human_pose(inspection_pose_quat)
    if rm.is_expert:
        rm.set_desired_rot_offset(desired_rot_offset)

    # Learn weights
    start_time = time.time()
    if max_adaptation_time_sec is None:
        max_adaptation_time_sec = 1e10  # will still stop after max iters
    for update_step in range(100):
        # print("UPDATE STEP: ", update_step)
        if max_adaptation_time_sec is not None and time.time() - start_time > max_adaptation_time_sec:
            break
        rand_idx = np.random.randint(0, num_perturbs)
        rm.update_weights_one_step(
        expert=all_expert_pose_traj[rand_idx], method="...",
        goal_pose_euler=goal_pose_euler)
        
    np.savez(os.path.join(save_folder, "online_weights.npz"),
        pos_weights=rm.pos_weights,
        ori_weights=rm.ori_weights
    )



if __name__ == "__main__":
    ########################################################
    args = parse_arguments()
    argparse_dict = vars(args)

    # define save path
    save_folder = f"exp1/online_is_expert_{args.is_expert}_saved_trials_inspection/eval_perturbs_{args.num_perturbs}_time_{args.max_adaptation_time_sec}"
    os.makedirs(save_folder, exist_ok=True)

    # Instead of loading trained model, we train on the fly here so we
    # can plot performance vs adaptation time
    # NOTE: USE UNIFIED's perturbation to train weights
    load_folder = f"exp1/unified_saved_trials_inspection/perturb_collection"

    # Define reward model
    state_dim = 3 + 3  # (pos, rot euler)
    # context: human pose
    context_dim = 3 + 3
    input_dim = state_dim + context_dim  # robot pose, human pose

    # is_expert: use apriori knowledge of human and dist to humna
    rm = PredefinedReward(is_expert=args.is_expert)

    # -> (left-mult) inv(R_inspection) * R_perturb = R_desired_offset
    # You can verify this makes sense by plotting with plot_ee_traj.py
    # which shows the estimatedd desired orientation
    inspection_ori_quat_from_perturb = inspection_ori_quats[0]
    inspection_pos_from_perturb = inspection_poses[0]
    orig_perturb_traj = np.load("/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/perturb_collection/perturb_traj_iter_0_num_0.npy")
    desired_rot_offset = (
        R.from_quat(inspection_ori_quat_from_perturb).inv() *
        R.from_quat(orig_perturb_traj[-1, 3:])).as_quat()

    # rm.load(folder=save_folder, name="online_weights.npz")
    # # desired rot is just the perturb ori initially
    # import ipdb
    # ipdb.set_trace()
    # rm.set_desired_pose(inspection_pos_from_perturb, inspection_ori_quat_from_perturb)

    run_adaptation(rm, desired_rot_offset, save_folder, collected_folder=load_folder, num_perturbs=args.num_perturbs,
                   max_adaptation_time_sec=args.max_adaptation_time_sec)

    it = 0
    pose_error_tol = 0.1
    max_pose_error_tol = 0.2  # too far to be considered converged and not moving
    del_pose_tol = 0.005  # over del_pose_interval iterations
    num_exps = len(start_poses)
    # num_exps = 3
    num_rand_trials = 5
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
        inspection_pos = inspection_poses[exp_iter]
        inspection_ori_quat = inspection_ori_quats[exp_iter]
        inspection_ori_euler = R.from_quat(inspection_ori_quat).as_euler("XYZ")
        inspection_pose_euler = np.concatenate(
            [inspection_pos, inspection_ori_euler])

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
            inspection_pose_noisy = np.concatenate([
                inspection_pose_euler[0:3] + rand_pos_noise(),
                inspection_pose_euler[3:] + rand_rot_euler_noise()
            ])
            inspection_pose_noisy_quat = np.concatenate([
                inspection_pose_noisy[0:3],
                R.from_euler("XYZ", inspection_pose_noisy[3:]).as_quat()
            ])

            # setting human pose
            rm.set_human_pose(inspection_pose_noisy_quat)
            
            trajopt = TrajOptExp(home=start_pose_noisy,
                                goal=goal_pose_noisy,
                                human_pose_euler=inspection_pose_noisy,
                                context_dim=context_dim,
                                use_state_features=False,
                                waypoints=TRAJ_LEN, max_iter=300)
            traj = Trajectory(
                waypts=trajopt.optimize(
                    context=inspection_pose_noisy, reward_model=rm),
                waypts_time=waypts_time)

            np.savez(os.path.join(save_folder, f"ee_pose_traj_iter_{exp_iter}_rand_trial_{rand_trial}.npz"), traj=traj.waypts, start_pose=start_pose_noisy, goal_pose=goal_pose_noisy, inspection_pose=inspection_pose_noisy)

            pbar.update(1)
            # exit()

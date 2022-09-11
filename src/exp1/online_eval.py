		
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


class PredefinedReward(object):
    def __init__(self) -> None:
        # some poses to either avoid or attract to
        self.positions = []
        self.orientations = []  # quaternions
        self.pos_weights = np.array([])  # to be optimized
        self.ori_weights = np.array([])  # to be optimized
        pass

    def dist(self, traj, pos):
        return np.linalg.norm(traj[:, 0:3] - pos, axis=-1).sum()

    def ori_dist(self, traj, ori_quat):
        traj_quat = R.from_euler("XYZ", traj[:, 3:]).as_quat()
        ori_quat = R.from_euler("XYZ", ori_quat).as_quat()
        return np.arccos(np.abs(traj_quat @ ori_quat)).sum()

    def reward(self, x, ret_single_value=True):
        traj = x
        pos_dists = np.array([self.dist(traj, pos) for pos in self.positions])
        ori_dists = np.array([self.ori_dist(traj, ori) for ori in self.orientations])

        if ret_single_value:
            return -1 * (self.pos_weights @ pos_dists + 
                     self.ori_weights @ ori_dists)
        else:
            return np.concatenate([pos_dists, ori_dists])

    def learn_weights(self, orig, expert):
        orig_feats = self.reward(orig, ret_single_value=False)
        expert_feats = self.reward(expert, ret_single_value=False)
        update = expert_feats - orig_feats


def OracleReward(PredefinedReward):
    def __init__(self, human_pose):
        self.positions = [human_pose[0:3]]
        self.orientations = [human_pose[3:]]
        self.weights = np.ones(1)

def MissingReward(PredefinedReward):
    def __init__(self):
        PredefinedReward.__init__(self)
        # some random positions of random objects
        self.positions = [
            np.array([0.2, 0.35, -0.08]),
            np.array([0.6, -0.5, 0.1]),
            np.array([-0.6, 0.2, 0.2]),
            np.array([0.3, -0.1, 0.1]),
        ]
        # discrete sampling of some rotations
        for rx in range(0, 360+1, 30):
            for ry in range(0, 360+1, 30):
                for rz in range(0, 360+1, 30):
                    self.orientations.append(
                        R.from_euler("XYZ", [rx,ry,rz], degrees=True).as_quat()
                    )

        self.weights = np.ones(len(self.positions))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store',
                        type=str, help="trained model name", default="policy_3D")
    parser.add_argument('--loaded_epoch', action='store',
                        type=int, default=100)
    parser.add_argument('--view_ros', action='store_true',
                        help="visualize 3D scene with ROS Rviz")
    parser.add_argument('--collected_folder', action='store', type=str)
    parser.add_argument('--max_adaptation_time_sec',
                        action='store', type=float)
    parser.add_argument('--num_perturbs', action='store',
                        type=int, required=True)
    parser.add_argument('--is_expert', action='store_true',
                        help="use expert, true features")
    args = parser.parse_args()

    return args


def run_adaptation(collected_folder, num_perturbs, max_adaptation_time_sec):
    files = os.listdir(collected_folder)
    exp_iter = None
    all_perturb_pose_traj = []
    for f in files:
        matches = re.findall("perturb_traj_iter_(\d+)_num_\d+.npy", f)
        if len(matches) > 0:
            exp_iter = int(matches[0])

            if len(all_perturb_pose_traj) < num_perturbs:
                perturb_pose_traj_quat = np.load(os.path.join(
                    collected_folder, f))

                perturb_pose_traj_euler = np.hstack([
                    perturb_pose_traj_quat[:, 0:3],
                    R.from_quat(perturb_pose_traj_quat[:, 3:]).as_euler("XYZ")
                ])

                all_perturb_pose_traj.append(perturb_pose_traj_euler)

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

    start_time = time.time()

    # Data processing
    all_processed_perturb_pose_traj = []
    num_wpts = 40
    for perturb_pose_traj_euler in all_perturb_pose_traj:

        waypts_time=np.linspace(0, num_wpts, len(perturb_pose_traj_euler))
        perturb_pose_traj_euler = Trajectory(
                waypts=perturb_pose_traj_euler,
                waypts_time=waypts_time
            ).downsample(num_waypts=num_wpts).waypts

        all_processed_perturb_pose_traj.append(perturb_pose_traj_euler)

        # Generate the original trajectories starting from the perturbation pose
        # that will be used to compare with the perturbation trajectories
        initial_pose = perturb_pose_traj_euler[0]
        traj = Trajectory(waypts=trajopt.optimize(
                context=inspection_pose_euler, reward_model=rm1),
            waypts_time=waypts_time)

    num_deforms = 1000
    deformed = []
    demo_idxs = []
    for _ in tqdm(range(num_deforms)):
        rand_demo_idx = np.random.randint(0, num_perturbs)
        perturb_pose_traj_euler = all_processed_perturb_pose_traj[rand_demo_idx]
        rand_noise = np.random.normal(0, scale=rm1.noise, size=rm1.action_dim)
        demo_deformed = rm1.deform(perturb_pose_traj_euler, rand_noise)
        demo_deformed_tensor = torch.from_numpy(
            demo_deformed).to(DEVICE).to(torch.float32)
        deformed.append(demo_deformed_tensor)
        demo_idxs.append(rand_demo_idx)  # need the original demo idx to compare with
    print("generating deformed trajectories... DONE!")

    # Perform adaptation
    context = inspection_pose_euler[np.newaxis, :].repeat(
        num_wpts, axis=0)

    # include initial data processing in adaptation time
    if max_adaptation_time_sec is not None:
        max_adaptation_time_sec -= (time.time() - start_time)
    else:
        max_adaptation_time_sec = 1e10  # will still stop after max iters
    rm.train_rewards(deformed=deformed, demo_idxs=demo_idxs, demos=all_processed_perturb_pose_traj,
                     context=context, max_time=max_adaptation_time_sec)

    # traj_deform = pertturb_pose_traj
    # new_features = self.environment.featurize(traj_deform.waypts)
    # old_features = self.environment.featurize(traj.waypts)
    # Phi_p = np.array([sum(x) for x in new_features])  # sum over waypoints
    # Phi = np.array([sum(x) for x in old_features])
    # update = Phi_p - Phi

    # if self.feat_method == "all":
    #     # Update all weights. 
    #     curr_weight = self.all_update(update)
    # elif self.feat_method == "max":
    #     # Update only weight of maximal change.
    #     curr_weight = self.max_update(update)
    # elif self.feat_method == "beta":
    #     # Confidence matters. Update weights with it.
    #     curr_weight = self.confidence_update(update, betas)
    # else:
    #     raise Exception('Learning method {} not implemented.'.format(self.feat_method))

    print "Here is the update:", update
    print "Here are the old weights:", self.environment.weights
    print "Here are the new weights:", curr_weight
    self.environment.weights = np.maximum(curr_weight, np.zeros(curr_weight.shape))

if __name__ == "__main__":
    ########################################################
    args = parse_arguments()
    argparse_dict = vars(args)

    # define save path
    save_folder = f"exp1/ferl_saved_trials_inspection/eval_perturbs_{args.num_perturbs}_time_{args.max_adaptation_time_sec}"
    os.makedirs(save_folder, exist_ok=True)

    # Instead of loading trained model, we train on the fly here so we
    # can plot performance vs adaptation time
    load_folder = f"exp1/unified_saved_trials_inspection/perturb_collection"

    # Define reward model
    state_dim = 3 + 3  # (pos, rot euler)
    # context: human pose
    context_dim = 1 + 3 + 3 if args.use_state_features else 3 + 3
    input_dim = state_dim + context_dim  # robot pose, human pose
    rm1 = TrainReward(model_dim=(input_dim, 128),
                      epoch=2000, traj_len=TRAJ_LEN, device=DEVICE)

    # rm1.load(folder=load_folder, name="exp_0_adapt_iter_0")
    run_adaptation(rm1, collected_folder=load_folder, num_perturbs=args.num_perturbs,
                   max_adaptation_time_sec=args.max_adaptation_time_sec)

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

        # traj = traj.waypts.copy()
        # traj = np.hstack([
        #     traj[:, 0:3],
        #     R.from_euler("XYZ", traj[:, 3:]).as_quat()
        # ])
        # np.save(
        #     f"{save_folder}/ee_pose_traj_iter_{0}_rand_trial_{0}.npy", traj)
        # exit()

        for rand_trial in range(10):
            # initialize target pose variables
            cur_pos = np.copy(start_pose[0:3])
            cur_ori_euler = np.copy(start_pose[3:])

            intervene_count = 0
            pose_error = 1e10
            del_pose = 1e10
            del_pose_running_avg = RunningAverage(length=5, init_vals=1e10)

            ee_pose_traj = []
            prev_pose_quat = None
            step = 0
            max_steps = 100
            dt = 0.5
            while (pose_error > pose_error_tol and
                    (pose_error > max_pose_error_tol) and step < max_steps):
                step += 1
                # calculate next action to take based on planned traj
                cur_pose = np.concatenate([cur_pos, cur_ori_euler])
                cur_ori_quat = R.from_euler("XYZ", cur_ori_euler).as_quat()
                cur_pose_quat = np.concatenate([cur_pos, cur_ori_quat])
                pose_error = calc_pose_error(
                    goal_pose_quat, cur_pose_quat, rot_scale=0)

                ee_pose_traj.append(cur_pose_quat.copy())

                local_target_pose = traj.interpolate(
                    t=step * dt).flatten()
                local_target_pos = local_target_pose[0:3]
                local_target_ori_quat = R.from_euler(
                    "XYZ", local_target_pose[3:]).as_quat()

                # 0 mean pos_std noise
                pos_std = 0.05
                rot_euler_std = 5 * np.pi / 180
                pos_noise = np.random.normal(loc=0, scale=pos_std, size=3)
                rot_noise = np.random.normal(
                    loc=0, scale=rot_euler_std, size=3)

                cur_pos = local_target_pos + pos_noise
                cur_ori_euler = local_target_pose[3:] + rot_noise

                print("dist_to_goal: ", pose_error)
                prev_pose_quat = np.copy(cur_pose_quat)

            # Save robot traj and intervene traj
            np.save(
                f"{save_folder}/ee_pose_traj_iter_{exp_iter}_rand_trial_{rand_trial}.npy", ee_pose_traj)

            print("Finished!")
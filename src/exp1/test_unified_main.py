import numpy as np
import argparse
import os
from scipy.spatial.transform import Rotation as R
import time
import signal

import rospy

from globals import *
add_paths()  # sets the paths for the below imports

from exp_utils import *

from unified_trajopt import TrajOptBase
from unified_reward import TrainReward
from trajectory import Trajectory


signal.signal(signal.SIGINT, sigint_handler)


"""
An extremely modified version of unified_main.py.
unified_main.py preserved the original logic of "test_main.py",
but just cleaned up bad coding practices that could lead
to bugs later.

NOTE: because image/video results not necessary, no need
for approach pose, grasp pose, grabbing object, etc..
just move EE

NOTE: we could make this cleaner by defining generic objects, 
generic start/goals that are read from an experiment-specific file,
but we don't have many experiments, so just copy all code to separate python
files.

NOTE: temporarily don't pick things up just to speed up testing,
then at the end we can add it back for final pictures if necessary

This file changes the logic heavily:
- no pre-loaded demo.pkl files
- no query/comparisons, only physical corrections
- pick up boxes
- allow for OPA state features to be used instead of raw object state 
for fair comparison with our approach since this could simplify learning
"""

# Hyperparameters
TRAJ_LEN = 10  # fixed number of wayponts for trajopt
waypts_time = np.linspace(0, T, TRAJ_LEN)
adapt_num_epochs = 5

DEVICE = "cpu"
print("DEVICE: %s" % DEVICE)

# Constants
reward_model1_path = "models/reward_model_1"
EXP_TYPE = "exp1"

DEBUG = False
if DEBUG:
    dstep = 0.05
    ros_delay = 0.1
else:
    dstep = 0.12
    ros_delay = 0.4  # NOTE: if modify this, must also modify rolling avg window of dpose


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

    def trajcost_true(self, xi):
        xi = xi.reshape(self.n_waypoints, self.state_dim)
        R = 0
        for idx in range(self.n_waypoints):
            # move closer to human
            reward_pos = - \
                np.linalg.norm(self.human_pose_quat[0:3] - xi[idx, 0:3])

            # present item rotated correctly
            reward_rot = -np.arccos(
                np.abs(self.desired_present_item_rot @ xi[idx, 3:7]))

            R += self.POS_WEIGHT * reward_pos + self.ROT_WEIGHT * reward_rot

        cost = -R
        return cost


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_state_features', action='store_true',
                        help="context are the same state features that OPA uses, not raw object state")
    parser.add_argument('--view_ros', action='store_true',
                        help="visualize 3D scene with ROS Rviz")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    # Define reward model
    state_dim = 3 + 3  # (pos, rot euler)
    # context: human pose
    context_dim = 1 + 3 + 3 if args.use_state_features else 3 + 3
    input_dim = state_dim + context_dim  # robot pose, human pose
    rm1 = TrainReward(noise=0.2, model_dim=(input_dim, 128),
                      epoch=500, traj_len=TRAJ_LEN, device=DEVICE)

    adapt_iter = 0
    pose_error_tol = 0.1
    max_pose_error_tol = 0.2  # too far to be considered converged and not moving
    del_pose_tol = 0.005  # over del_pose_interval iterations
    perturb_pose_traj = [None] * TRAJ_LEN
    override_pred_delay = False
    num_exps = len(start_poses)
    for exp_iter in range(num_exps):
        # set extra mass of object to pick up
        # exp_iter = num_exps - 1
        exp_iter = min(exp_iter, num_exps - 1)
        extra_mass = extra_masses[exp_iter]

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

        # 1st iter: show default behavior
        # 2nd iter: perform adaptation on saved perturbation data
        # 3rd iter: visualize new behavior after adaptation
        if exp_iter == 1:
            perturb_pose_traj = np.load("unified_perturb_pose_traj.npy")
            perturb_pose_traj = Trajectory(
                waypts=perturb_pose_traj,
                waypts_time=np.linspace(0, 10, len(perturb_pose_traj))).downsample(num_waypts=20).waypts
            perturb_pose_traj = np.vstack(perturb_pose_traj)
            perturb_pose_traj_euler = np.hstack([
                perturb_pose_traj[:, 0:3],
                R.from_quat(perturb_pose_traj[:, 3:]).as_euler("XYZ")
            ])

            # Perform adaptation and re-run trajopt
            T = perturb_pose_traj.shape[0]
            context = inspection_pose_euler[np.newaxis, :].repeat(T, axis=0)
            rm1.train_rewards([perturb_pose_traj_euler, ], context=context)

        trajopt = TrajOptExp(home=start_pose,
                             goal=goal_pose,
                             human_pose_euler=inspection_pose_euler,
                             context_dim=context_dim,
                             use_state_features=args.use_state_features,
                             waypoints=len(perturb_pose_traj))
        traj = Trajectory(
            waypts=trajopt.optimize(
                context=inspection_pose_euler, reward_model=rm1),
            waypts_time=waypts_time)

        np.save(f"test_unified_traj_{exp_iter}.npy", traj.waypts)

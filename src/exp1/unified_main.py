import numpy as np
import argparse
import os
from scipy.spatial.transform import Rotation as R
import time

import rospy

from globals import *
from exp_utils import *
from kinova_interface import KinovaInterface

from unified_trajopt import TrajOptBase
from unified_reward import TrainReward
from trajectory import Trajectory

import signal

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
    kinova = KinovaInterface()

    # Start of unified-learning-specific code
    # define save path
    trial_num = len(os.listdir("unified_saved_trials"))
    save_folder = f"unified_saved_trials/trial_{trial_num}"
    os.makedirs(save_folder)

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
    perturb_pose_traj = []
    override_pred_delay = False
    num_exps = len(start_poses)
    for exp_iter in range(num_exps + 1):  # +1 for original start pos
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
        inspection_pose = np.concatenate(
            [inspection_pos, inspection_ori_euler])

        if not DEBUG:
            kinova.reach_start_joints(HOME_JOINTS)

        approach_pose = np.copy(start_pose_quat)
        approach_pose[2] += 0.2
        kinova.reach_start_pos(approach_pose, goal_pose_quat, [], [])
        if item_ids[exp_iter] == BOX_ID:
            approach_pose_v2 = np.copy(start_pose_quat)
            approach_pose_v2[1] -= 0.1
            kinova.reach_start_pos(
                approach_pose_v2, goal_pose_quat, [], [])
        if item_ids[exp_iter] == CAN_ID:
            approach_pose_v2 = np.copy(start_pose_quat)
            approach_pose_v2[0] -= 0.1
            approach_pose_v2[1] -= 0.03
            kinova.reach_start_pos(
                approach_pose_v2, goal_pose_quat, [], [])
        kinova.reach_start_pos(start_pose_quat, goal_pose_quat, [], [])

        trajopt = TrajOptExp(home=start_pose,
                             goal=goal_pose,
                             human_pose_euler=inspection_pose,
                             context_dim=context_dim,
                             use_state_features=args.use_state_features,
                             waypoints=TRAJ_LEN)
        traj = Trajectory(
            waypts=trajopt.optimize(context=inspection_pose, reward_model=rm1),
            waypts_time=waypts_time)
        local_target_pos = traj.waypts[0, 0:3]
        local_target_ori_quat = R.from_euler(
            "XYZ", traj.waypts[0, 3:]).as_quat()

        intervene_count = 0
        pose_error = 1e10
        del_pose = 1e10
        del_pose_running_avg = RunningAverage(length=5, init_vals=1e10)

        ee_pose_traj = []
        is_intervene_traj = []

        prev_pose_quat = None
        # modified MPC Fashion: run trajopt every K steps
        K = 10
        step = 0
        start_t = time.time()
        while (not rospy.is_shutdown() and pose_error > pose_error_tol and
                (pose_error > max_pose_error_tol)):
            step += 1
            # if step % K == 0:
            #     # run trajopt in MPC fashion, new traj starting from current pose
            #     trajopt = TrajOptExp(home=cur_pose,
            #                          goal=goal_pose,
            #                          human_pose_euler=inspection_pose,
            #                          context_dim=context_dim,
            #                          waypoints=TRAJ_LEN)
            #     traj = Trajectory(
            #         waypts=trajopt.optimize(
            #             context=inspection_pose, reward_model=rm1),
            #         waypts_time=waypts_time)
            #     start_t = time.time()

            # calculate next action to take based on planned traj
            cur_pose_quat = np.concatenate(
                [kinova.cur_pos, kinova.cur_ori_quat])
            cur_pose = np.concatenate([kinova.cur_pos,
                                       R.from_quat(kinova.cur_ori_quat).as_euler("XYZ")])
            pose_error = calc_pose_error(
                goal_pose_quat, cur_pose_quat, rot_scale=0)
            if prev_pose_quat is not None:
                del_pose = calc_pose_error(prev_pose_quat, cur_pose_quat)
                del_pose_running_avg.update(del_pose)

            ee_pose_traj.append(cur_pose_quat.copy())
            is_intervene_traj.append(kinova.is_intervene)

            kinova.perturb_pose_traj = np.load("unified_perturb_pose_traj.npy")

            if kinova.need_update and not DEBUG:
                # Hold current pose while running adaptation
                for i in range(5):
                    kinova.pose_pub.publish(pose_to_msg(
                        cur_pose_quat, frame=ROBOT_FRAME))
                rospy.sleep(0.1)
                kinova.is_intervene = False
                kinova.publish_is_intervene()

                assert len(
                    kinova.perturb_pose_traj) > 1, "Need intervention traj of > 1 steps"

                # TODO: for fair comparison, should we still use the assumption that human perturb is linear?
                # dist = np.linalg.norm(
                #     kinova.perturb_pose_traj[-1][0:POS_DIM] - kinova.perturb_pose_traj[0][0:POS_DIM])
                # 1 step for start, 1 step for goal at least
                # T = max(2, int(np.ceil(dist / dstep)))
                # T = 5
                # perturb_pos_traj_interp = np.linspace(
                #     start=kinova.perturb_pose_traj[0][0:POS_DIM], stop=kinova.perturb_pose_traj[-1][0:POS_DIM], num=T)
                # final_perturb_ori = kinova.perturb_pose_traj[-1][POS_DIM:]

                # perturb_ori_traj = np.copy(final_perturb_ori)[
                #     np.newaxis, :].repeat(T, axis=0)
                # perturb_pos_traj = perturb_pos_traj_interp
                # perturb_pose_traj = np.hstack(
                #     [np.vstack(perturb_pos_traj), perturb_ori_traj])

                perturb_pose_traj = np.vstack(kinova.perturb_pose_traj)
                perturb_pose_traj_euler = np.concatenate([
                    perturb_pose_traj[:, 0:3],
                    R.from_quat(perturb_pose_traj[:, 3:]).as_euler("XYZ")
                ])

                # Perform adaptation and re-run trajopt
                rm1.train_rewards(perturb_pose_traj_euler)

                # Save adapted reward model
                rm1.save(folder=save_folder,
                         name=f"exp_{exp_iter}_adapt_iter_{adapt_iter}")
                adapt_iter += 1

                # Re-run trajopt at final, perturbed state
                trajopt = TrajOptExp(home=cur_pose,
                                     goal=goal_pose,
                                     human_pose_euler=inspection_pose,
                                     context_dim=context_dim,
                                     waypoints=TRAJ_LEN)
                traj = Trajectory(
                    waypts=trajopt.optimize(
                        context=inspection_pose, reward_model=rm1),
                    waypts_time=waypts_time)
                start_t = time.time()

                # reset the intervention data
                kinova.perturb_pose_traj = []
                kinova.need_update = False
                override_pred_delay = True

                # increment intervention count to avoid overwriting this intervention's data
                intervene_count += 1

                # reach back to the pose before p
                continue

            elif step % 2 == 0 or override_pred_delay:
                print("new target", step)
                # calculate new action
                override_pred_delay = False
                local_target_pose = traj.interpolate(
                    t=time.time() - start_t).flatten()
                local_target_pos = local_target_pose[0:3]
                local_target_ori_quat = R.from_euler(
                    "XYZ", local_target_pose[3:]).as_quat()

            # TODO: are these necessary, or can we move this into
            # traj opt with constraints and cost
            # Apply low-pass filter to smooth out policy's sudden changes in orientation
            interp_rot = interpolate_rotations(
                start_quat=kinova.cur_ori_quat, stop_quat=local_target_ori_quat, alpha=0.7)

            # Clip target EE position to bounds
            local_target_pos = np.clip(
                local_target_pos, a_min=kinova.ee_min_pos, a_max=kinova.ee_max_pos)

            # Publish is_intervene
            kinova.publish_is_intervene()

            # Publish target pose
            if not DEBUG:
                target_pose = np.concatenate(
                    [local_target_pos, interp_rot])
                kinova.pose_pub.publish(pose_to_msg(
                    target_pose, frame=ROBOT_FRAME))

            if DEBUG:
                kinova.cur_pos = local_target_pos
                kinova.cur_ori_quat = interp_rot

            if step % 2 == 0:
                print("Pos error: ", np.linalg.norm(
                    local_target_pos - kinova.cur_pos))
                print("Ori error: ", np.linalg.norm(
                    np.arccos(np.abs(kinova.cur_ori_quat @ local_target_ori_quat))))
                print("Dpose: ", del_pose_running_avg.avg)
                print()

            prev_pose_quat = np.copy(cur_pose_quat)
            rospy.sleep(0.3)

        # Save robot traj and intervene traj
        np.save(f"{save_folder}/ee_pose_traj_iter_{exp_iter}.npy", ee_pose_traj)
        np.save(f"{save_folder}/is_intervene_traj{exp_iter}.npy",
                is_intervene_traj)

        print(
            f"Finished! Error {pose_error} vs tol {pose_error_tol}, \nderror {del_pose_running_avg.avg} vs tol {del_pose_tol}")
        print("Opening gripper to release item")

        rospy.sleep(0.1)

import numpy as np
import argparse
import os
from scipy.spatial.transform import Rotation as R
import time
import signal
import re

import rospy

from globals import *
add_paths()  # sets the paths for the below imports

from exp_utils import *
from kinova_interface import KinovaInterface

from unified_trajopt import TrajOptBase
from unified_reward import TrainReward
from trajectory import Trajectory


signal.signal(signal.SIGINT, sigint_handler)


# Hyperparameters
TRAJ_LEN = 10  # fixed number of wayponts for trajopt
waypts_time = np.linspace(0, T, TRAJ_LEN)
adapt_num_epochs = 5

DEVICE = "cpu"
print("DEVICE: %s" % DEVICE)

# Constants
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


def load_pose_as_quat(pose):
    if pose.shape[-1] == 7:
        return pose
    elif pose.shape[-1] == 6:
        print("CONVERTING EULER TO QUAT...")
        if len(pose.shape) > 1:
            return np.hstack([pose[..., 0:3], R.from_euler("XYZ", pose[..., 3:]).as_quat()])
        else:
            return np.concatenate([pose[0:3], R.from_euler("XYZ", pose[3:]).as_quat()])
    else:
        raise Exception("Unexpected pose shape! ", pose.shape)


def reach_start_joints(viz_3D_publisher, target_joints, joint_error_tol=0.1, dpose_tol=1e-2):
    global cur_pos_world, cur_ori_quat, cur_joints
    global joints_lb, joints_ub, joints_avoid_wraparound

    dEE_pos = 1e10
    dEE_pos_running_avg = RunningAverage(length=5, init_vals=1e10)
    joint_error = 1e10
    joint_error_tol = 0.01
    max_delta_joints = np.deg2rad([5, 5, 5, 5, 10, 20, 20])
    djoints = np.zeros(len(cur_joints))
    prev_pos_world = None
    while not rospy.is_shutdown() and (
            cur_pos_world is None or cur_joints is None or
            (joint_error > joint_error_tol and dEE_pos_running_avg.avg > dpose_tol)):
        # for each joint, linearly interpolate with a max change
        # interpolate such that joints don't cross over joint limits
        for ji in range(len(cur_joints)):
            if joints_avoid_wraparound[ji]:
                djoints[ji] = target_joints[ji] - cur_joints[ji]
            else:
                # find shortest direction, allowing for wraparound
                # ex: target = 3pi/4, cur = -3pi/4, djoint = normalize(6pi/4) = -2pi/4
                djoints[ji] = normalize_pi_neg_pi(
                    target_joints[ji] - cur_joints[ji])

        # calculate joint error
        joint_error = np.abs(djoints).sum()

        # clip max change
        print("cur joints: ", cur_joints)
        print("target joints: ", target_joints)
        print("djoints before: ", djoints)
        djoints = np.clip(djoints, a_min=-max_delta_joints,
                          a_max=max_delta_joints)
        print("djoints after: ", djoints)

        # publish target joint
        joints_deg_pub.publish(Float64MultiArray(
            data=np.rad2deg(cur_joints + djoints)))
        print("Reaching target joints...")
        print("joint error (%.3f) dpos: (%.3f)" % (
            joint_error, dEE_pos_running_avg.avg))

        # calculate pose change
        if prev_pos_world is not None:
            dEE_pos = np.linalg.norm(cur_pos_world - prev_pos_world)
        dEE_pos_running_avg.update(dEE_pos)
        prev_pos_world = np.copy(cur_pos_world)

        all_objects = [
            Object(
                pos=cur_pos_world, radius=Net2World * Params.agent_radius, ori=cur_ori_quat)
        ]
        viz_3D_publisher.publish(objects=all_objects, object_colors=[
                                 Params.agent_color_rgb, ])
        rospy.sleep(ros_delay)

    print("Final joint error: ", joint_error)


def reach_start_pos(pose_pub, start_pose, goal_pose,
                    pose_tol=0.03, dpose_tol=1e-2, reaching_dstep=0.02, clip_movement=False):
    global cur_pos_world, cur_ori_quat, cur_joints

    if DEBUG:
        cur_pos_world = np.copy(start_pose[0:3])
        cur_ori_quat = np.copy(start_pose[3:])
        # cur_pos_world = np.copy(inspection_pos_world)
        # cur_ori_quat = np.copy(inspection_ori_quat)
        return

    else:
        start_dist = np.linalg.norm(start_pose[0:3] - cur_pos_world)
        pose_error = 1e10
        dEE_pos = 1e10
        dEE_pos_running_avg = RunningAverage(length=5, init_vals=1e10)
        prev_pos_world = None
        while not rospy.is_shutdown() and (
                cur_pos_world is None or (pose_error > pose_tol and dEE_pos_running_avg.avg > dpose_tol)):
            pos_vec = start_pose[0:3] - cur_pos_world
            # pos_vec[2] = np.clip(pos_vec[2], -0.06, 0.1)
            pos_mag = np.linalg.norm(pos_vec)
            pos_vec = pos_vec * min(pos_mag, reaching_dstep) / pos_mag
            # translation along certain directions involve certain joints which can be larger or smaller
            # apply limits to horizontal movement to prevent 0th joint from rotating too fast
            # print("pos_vec before: ", pos_vec)
            if clip_movement:
                pos_vec = np.clip(
                    pos_vec, a_min=[-0.07, -0.07, -0.1], a_max=[0.07, 0.07, 0.1])
            # print("pos_vec after: ", pos_vec)
            target_pos_world = cur_pos_world + pos_vec
            # target_pos_world = cur_pos_world + np.array([0, -0.1, 0.2])
            dist_to_start_ratio = min(pos_mag / (start_dist + 1e-5), 1.0)
            target_ori_quat = interpolate_rotations(start_quat=cur_ori_quat, stop_quat=start_pose[3:],
                                                    alpha=1 - dist_to_start_ratio)
            pose_pub.publish(
                pose_to_msg(np.concatenate([target_pos_world, target_ori_quat]), frame=ROBOT_FRAME))

            if cur_pos_world is None:
                print("Waiting to receive robot pos")
            else:
                cur_pose = np.concatenate([cur_pos_world, cur_ori_quat])
                pose_error = calc_pose_error(
                    cur_pose, start_pose, rot_scale=0.1)
                if prev_pos_world is not None:
                    dEE_pos = np.linalg.norm(cur_pos_world - prev_pos_world)
                dEE_pos_running_avg.update(dEE_pos)
                prev_pos_world = np.copy(cur_pos_world)
                print("Waiting to reach start pos (%s), cur pos (%s) error: %.3f,  change: %.3f" %
                      (np.array2string(target_pos_world, precision=2), np.array2string(cur_pos_world, precision=2),
                       pose_error, dEE_pos_running_avg.avg))
            rospy.sleep(ros_delay)
        rospy.sleep(0.5)  # pause to let arm finish converging

        print("Final error: ", pose_error)
        print(cur_pos_world, start_pose[0:3])
        print("Cur joints: ", cur_joints)




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_traj_path', action='store',
                        type=str)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    kinova = KinovaInterface()

    perturb_iter = 0
    exp_iter = int(re.findall("ee_pose_traj_iter_(\d+)\S+", args.saved_traj_path)[0])
    traj_data = np.load(args.saved_traj_path)
    goal_pose_quat = goal_poses[exp_iter]
    start_pose_quat = start_poses[exp_iter]
    start_pose_world = traj_data['start_pose']
    if start_pose_world.shape[0] == 6:
        start_pose_world = np.hstack([start_pose_world[:3], R.from_euler("XYZ", start_pose_world[3:]).as_quat()])
    if traj_data['traj'].shape[1] == 6:
        saved_traj = np.hstack([traj_data['traj'][:,:3], R.from_euler("XYZ", traj_data['traj'][:,3:]).as_quat()])
    else:
        saved_traj = traj_data['traj']

    prev_pose_quat = None
    pose_error_tol = 0.2
    max_pose_error_tol = 0.2  # too far to be considered converged and not moving
    del_pose_tol = 0.005  # over del_pose_interval iterations
    step = 0
    pose_error = 1e10
    del_pose = 1e10
    # del_pose_running_avg = RunningAverage(length=5, init_vals=1e10)
    start_t = time.time()

    # move to start
    if not DEBUG:
        kinova.reach_joints(HOME_JOINTS)

    perform_grasp(start_pose_world, item_ids[0], kinova)

    for step in range(len(saved_traj)):
        print("STEP: ", step)
        local_target_pose = load_pose_as_quat(saved_traj[step])

        pose_error = 1e10
        del_pose = 1e10

        prev_pose_world = None
        max_inner_steps = 5
        inner_step = 0
        while (not rospy.is_shutdown() and pose_error > pose_error_tol and
                inner_step < max_inner_steps):
        
            # calculate next action to take based on planned traj
            cur_pose_quat = np.concatenate(
                [kinova.cur_pos, kinova.cur_ori_quat])
            cur_pose = np.concatenate([kinova.cur_pos,
                                        R.from_quat(kinova.cur_ori_quat).as_euler("XYZ")])
            pose_error = calc_pose_error(
                local_target_pose, cur_pose_quat, rot_scale=0)
            print("Local pose error: ", pose_error)

            # calculate new action
            local_target_pos = local_target_pose[0:3]
            local_target_ori_quat = local_target_pose[3:]

            # TODO: are these necessary, or can we move this into
            # traj opt with constraints and cost
            # Apply low-pass filter to smooth out policy's sudden changes in orientation
            interp_rot = interpolate_rotations(
                start_quat=kinova.cur_ori_quat, stop_quat=local_target_ori_quat, alpha=0.7)

            # # Clip target EE position to bounds
            # local_target_pos = np.clip(
            #     local_target_pos, a_min=kinova.ee_min_pos, a_max=kinova.ee_max_pos)
            # local_target_pos_clipped = np.clip(
            #     local_target_pos, a_min=kinova.ee_min_pos, a_max=kinova.ee_max_pos)
            # if (local_target_pos != local_target_pos_clipped).any():
            #     inner_step += 1
            #     continue

            # Publish target pose
            if not DEBUG:
                target_pose = np.concatenate(
                    [local_target_pos, interp_rot])
                # target_pose[:3] = 0.7 * target_pose[:3] + 0.3 * cur_pose[:3]
                kinova.pose_pub.publish(pose_to_msg(
                    target_pose, frame=ROBOT_FRAME))

            if DEBUG:
                kinova.cur_pos = local_target_pos
                kinova.cur_ori_quat = interp_rot

            prev_pose_quat = np.copy(cur_pose_quat)
            rospy.sleep(0.8)

            inner_step += 1

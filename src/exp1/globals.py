import numpy as np
import rospy
from pynput import keyboard
import torch

from viz_3D import msg_to_pose
from scipy.spatial.transform import Rotation as R

cuda = torch.cuda.is_available()
DEVICE = "cuda:0" if cuda else "cpu"
print("DEVICE: %s" % DEVICE)

num_objects = 1
ee_min_pos = np.array([0.23, -0.475, -0.1])
ee_max_pos = np.array([0.725, 0.55, 0.35])
HOME_POSE = np.array([0.373, 0.1, 0.13, 0.707, 0.707, 0, 0])
HOME_JOINTS = np.array([-0.705, 0.952, -1.663, -1.927, 2.131, 1.252, -0.438])
POS_DIM = 3
DROP_OFF_OFFSET = np.array([0.0, 0.0, -0.1])
T = 20.0  # taken from FERL yaml file

# only joints 1, 3, 5 need to avoid wraparound
joints_lb = np.array([-np.pi, -2.250, -np.pi, -2.580, -np.pi, -2.0943, -np.pi])
joints_ub = -1 * joints_lb
joints_avoid_wraparound = [False, True, False, True, False, True, False]
BOX_MASS = 0.3
CAN_MASS = 1.0

DEBUG = False
if DEBUG:
    dstep = 0.05
    ros_delay = 0.1
else:
    dstep = 0.12
    ros_delay = 0.4  # NOTE: if modify this, must also modify rolling avg window of dpose

# Item ID's (if item == can, slide into grasp pose horizontally)
BOX_ID = 0
CAN_ID = 1
item_ids = [
    BOX_ID,
    BOX_ID,
    BOX_ID,
]
# Item masses
BOX_MASS = 0.3
CAN_MASS = 1.0
extra_masses = [
    BOX_MASS,
    BOX_MASS,
    BOX_MASS,
]

# Item pickup poses
start_poses = [
    np.array([0.2, 0.35, -0.08]),
    np.array([0.2, 0.465, -0.08]),
    np.array([0.22, 0.58, -0.08]),
]
start_ori_quats = [
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),
    np.array([0, 1., 0, 0]),
]
# Item dropoff poses
goal_poses = [
    np.array([0.4, -0.475, 0.1]),
    np.array([0.38, -0.475, 0.1]),
    np.array([0.36, -0.475, 0.1]),
]
goal_ori_quats = start_ori_quats
inspection_poses = [
    np.array([0.7, 0.1, 0.05]),  # sitting at center of table
    np.array([0.6, -0.2, 0.4]),   # standing at corner of table
    np.array([0.6, -0.2, 0.4])   # TODO:
]
inspection_ori_quats = [
    R.from_euler("zyx", [0, 30, 0], degrees=True).as_quat(),
    R.from_euler("zyx", [-20, -20, 0], degrees=True).as_quat(),
    R.from_euler("zyx", [-20, -20, 0], degrees=True).as_quat()  # TODO:
]

# Global info updated by callbacks
cur_pos, cur_ori_quat = None, None
cur_joints = None
perturb_pose_traj = []
is_intervene = False
need_update = False


def is_intervene_cb(key):
    global is_intervene, need_update
    if key == keyboard.Key.space:
        is_intervene = not is_intervene
        if not is_intervene:  # True -> False, end of intervention
            need_update = True


def obj_pose_cb(msg):
    # TODO:
    global is_intervene
    if is_intervene:
        pass


def robot_pose_cb(msg):
    global is_intervene, cur_pos, cur_ori_quat, perturb_pose_traj, DEBUG
    pose = msg_to_pose(msg)
    if not DEBUG:
        cur_pos = pose[0:3]
        cur_ori_quat = pose[3:]
        if is_intervene:
            perturb_pose_traj.append(
                np.concatenate([cur_pos, cur_ori_quat]))


def robot_joints_cb(msg):
    global cur_joints
    cur_joints = np.deg2rad(msg.data)
    for i in range(len(cur_joints)):
        cur_joints[i] = normalize_pi_neg_pi(cur_joints[i])


def reach_start_joints(target_joints, joint_error_tol=0.1, dpose_tol=1e-2,
                       viz_3D_publisher=None):
    global cur_pos, cur_ori_quat, cur_joints
    global joints_lb, joints_ub, joints_avoid_wraparound

    dEE_pos = 1e10
    dEE_pos_running_avg = RunningAverage(length=5, init_vals=1e10)
    joint_error = 1e10
    joint_error_tol = 0.01
    max_delta_joints = np.deg2rad([5, 5, 5, 5, 10, 20, 20])
    djoints = np.zeros(len(cur_joints))
    prev_pos = None
    while not rospy.is_shutdown() and (
            cur_pos is None or cur_joints is None or
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

        # calculate spose change
        if prev_pos is not None:
            dEE_pos = np.linalg.norm(cur_pos - prev_pos)
        dEE_pos_running_avg.update(dEE_pos)
        prev_pos = np.copy(cur_pos)

        all_objects = [
            Object(
                pos=cur_pos, radius=Net2World * Params.agent_radius, ori=cur_ori_quat)
        ]
        if viz_3D_publisher is not None:
            viz_3D_publisher.publish(objects=all_objects, object_colors=[
                Params.agent_color_rgb, ])
        rospy.sleep(ros_delay)

    print("Final joint error: ", joint_error)


def reach_start_pos(start_pose, goal_pose, object_poses, object_radii,
                    pose_tol=0.03, dpose_tol=1e-2, reaching_dstep=0.1, clip_movement=False, viz_3D_publisher=None):
    global cur_pos, cur_ori_quat, cur_joints

    start_pos = Net2World * start_pose[:3]
    start_ori_quat = start_pose[3:]
    start_pose = np.concatenate([start_pos, start_ori_quat])
    if DEBUG:
        cur_pos = np.copy(start_pos)
        cur_ori_quat = np.copy(start_ori_quat)
        # cur_pos = np.copy(inspection_pos)
        # cur_ori_quat = np.copy(inspection_ori_quat)
        return

    else:
        start_dist = np.linalg.norm(start_pos - cur_pos)
        pose_error = 1e10
        dEE_pos = 1e10
        dEE_pos_running_avg = RunningAverage(length=5, init_vals=1e10)
        prev_pos = None
        while not rospy.is_shutdown() and (
                cur_pos is None or (pose_error > pose_tol and dEE_pos_running_avg.avg > dpose_tol)):
            pos_vec = start_pos - cur_pos
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
            target_pos = cur_pos + pos_vec
            # target_pos = cur_pos + np.array([0, -0.1, 0.2])
            dist_to_start_ratio = min(pos_mag / (start_dist + 1e-5), 1.0)
            target_ori_quat = interpolate_rotations(start_quat=cur_ori_quat, stop_quat=start_ori_quat,
                                                    alpha=1 - dist_to_start_ratio)
            pose_pub.publish(
                pose_to_msg(np.concatenate([target_pos, target_ori_quat]), frame=ROBOT_FRAME))
            is_intervene_pub.publish(False)

            # Publish objects
            object_colors = [
                (0, 255, 0),
            ]
            all_object_colors = object_colors + [
                Params.start_color_rgb,
                Params.goal_color_rgb,
                Params.agent_color_rgb,
            ]
            # force_colors = object_colors + [Params.goal_color_rgb]
            all_objects = [Object(pos=Net2World * pose[0:POS_DIM], ori=[0.7627784, -0.00479786, 0.6414479, 0.08179578],
                                  radius=Net2World * radius) for pose, radius in
                           zip(object_poses, object_radii)]
            all_objects += [
                Object(
                    pos=Net2World * start_pose[0:POS_DIM], radius=Net2World * Params.agent_radius, ori=start_pose[POS_DIM:]),
                Object(
                    pos=Net2World * goal_pose[0:POS_DIM], radius=Net2World * goal_rot_radius.item(), ori=goal_pose[POS_DIM:]),
                Object(
                    pos=cur_pos, radius=Net2World * Params.agent_radius, ori=cur_ori_quat)
            ]
            viz_3D_publisher.publish(
                objects=all_objects, object_colors=all_object_colors,)

            if cur_pos is None:
                print("Waiting to receive robot pos")
            else:
                cur_pose = np.concatenate([cur_pos, cur_ori_quat])
                pose_error = calc_pose_error(
                    cur_pose, start_pose, rot_scale=0.1)
                if prev_pos is not None:
                    dEE_pos = np.linalg.norm(cur_pos - prev_pos)
                dEE_pos_running_avg.update(dEE_pos)
                prev_pos = np.copy(cur_pos)
                print("Waiting to reach start pos (%s), cur pos (%s) error: %.3f,  change: %.3f" %
                      (np.array2string(target_pos, precision=2), np.array2string(cur_pos, precision=2),
                       pose_error, dEE_pos_running_avg.avg))
            rospy.sleep(ros_delay)
        rospy.sleep(0.5)  # pause to let arm finish converging

        print("Final error: ", pose_error)
        print(cur_pos, start_pos)
        print("Cur joints: ", cur_joints)

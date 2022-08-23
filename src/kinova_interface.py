import numpy as np
import rospy
from pynput import keyboard
import time

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, Float64MultiArray

from viz_3D import msg_to_pose
from scipy.spatial.transform import Rotation as R

from exp_utils import *


class KinovaInterface(object):
    def __init__(self, ros_node_name="kinova_policy", debug=False):
        rospy.init_node(ros_node_name)
        self.debug = debug

        # Global info updated by callbacks
        self.cur_pos, self.cur_ori_quat = None, None
        self.cur_joints = None
        self.perturb_pose_traj = []
        self.perturb_time_traj = []
        self.is_intervene = False
        self.need_update = False

        # Constants
        # EE bounds
        self.ee_min_pos = np.array([0.23, -0.475, -0.1])
        self.ee_max_pos = np.array([0.725, 0.55, 0.35])

        # Joint bounds
        # only joints 1, 3, 5 need to avoid wraparound
        self.joints_lb = np.array(
            [-np.pi, -2.250, -np.pi, -2.580, -np.pi, -2.0943, -np.pi])
        self.joints_ub = -1 * self.joints_lb
        self.joints_avoid_wraparound = [
            False, True, False, True, False, True, False]

        # ROS pubs/subs
        rospy.init_node('exp1_OPA')

        # Robot EE pose
        rospy.Subscriber('/kinova/pose_tool_in_base_fk',
                         PoseStamped, self.robot_pose_cb, queue_size=1)

        # Robot joint state
        rospy.Subscriber('/kinova/current_joint_state',
                         Float64MultiArray, self.robot_joints_cb, queue_size=1)

        # Target pose topic
        self.pose_pub = rospy.Publisher(
            "/kinova_demo/pose_cmd", PoseStamped, queue_size=10)

        # Target joints
        self.joints_deg_pub = rospy.Publisher(
            "/siemens_demo/joint_cmd", Float64MultiArray, queue_size=10)

        self.is_intervene_pub = rospy.Publisher(
            "/is_intervene", Bool, queue_size=10)
        self.gripper_pub = rospy.Publisher(
            "/siemens_demo/gripper_cmd", Bool, queue_size=10)
        self.extra_mass_pub = rospy.Publisher(
            "/gripper_extra_mass", Float32, queue_size=10)

        # Listen for keypresses marking start/stop of human intervention
        listener = keyboard.Listener(on_press=self.is_intervene_cb)
        listener.start()

    def is_intervene_cb(self, key):
        if key == keyboard.Key.space:
            self.is_intervene = not self.is_intervene
            if not self.is_intervene:  # True -> False, end of intervention
                self.need_update = True

    def obj_pose_cb(self, msg):
        if self.is_intervene:
            pass

    def robot_pose_cb(self, msg):
        pose = msg_to_pose(msg)
        if not self.debug:
            self.cur_pos = pose[0:3]
            self.cur_ori_quat = pose[3:]
            if self.is_intervene:
                self.perturb_pose_traj.append(
                    np.concatenate([self.cur_pos, self.cur_ori_quat]))
                self.perturb_time_traj.append(time.time())

    def robot_joints_cb(self, msg):
        self.cur_joints = np.deg2rad(msg.data)
        for i in range(len(self.cur_joints)):
            self.cur_joints[i] = normalize_pi_neg_pi(self.cur_joints[i])

    def reach_start_joints(self, target_joints, joint_error_tol=0.1, dpose_tol=1e-2,
                           viz_3D_publisher=None):

        dEE_pos = 1e10
        dEE_pos_running_avg = RunningAverage(length=5, init_vals=1e10)
        joint_error = 1e10
        joint_error_tol = 0.01
        max_delta_joints = np.deg2rad([5, 5, 5, 5, 10, 20, 20])
        djoints = np.zeros(7)
        prev_pos = None
        while not rospy.is_shutdown() and (
                self.cur_joints is None or self.cur_joints is None or
                (joint_error > joint_error_tol and dEE_pos_running_avg.avg > dpose_tol)):

            if self.cur_joints is None or self.cur_pos is None:
                print("Waiting to receive robot joints and pose")
                rospy.sleep(self.ros_delay)
                continue

            # for each joint, linearly interpolate with a max change
            # interpolate such that joints don't cross over joint limits
            for ji in range(len(self.cur_joints)):
                if self.joints_avoid_wraparound[ji]:
                    djoints[ji] = target_joints[ji] - self.cur_joints[ji]
                else:
                    # find shortest direction, allowing for wraparound
                    # ex: target = 3pi/4, cur = -3pi/4, djoint = normalize(6pi/4) = -2pi/4
                    djoints[ji] = normalize_pi_neg_pi(
                        target_joints[ji] - self.cur_joints[ji])

            # calculate joint error
            joint_error = np.abs(djoints).sum()

            # clip max change
            print("cur joints: ", self.cur_joints)
            print("target joints: ", target_joints)
            print("djoints before: ", djoints)
            djoints = np.clip(djoints, a_min=-max_delta_joints,
                              a_max=max_delta_joints)
            print("djoints after: ", djoints)

            # publish target joint
            self.joints_deg_pub.publish(Float64MultiArray(
                data=np.rad2deg(self.cur_joints + djoints)))
            print("Reaching target joints...")
            print("joint error (%.3f) dpos: (%.3f)" % (
                joint_error, dEE_pos_running_avg.avg))

            # calculate spose change
            if prev_pos is not None:
                dEE_pos = np.linalg.norm(self.cur_pos - prev_pos)
            dEE_pos_running_avg.update(dEE_pos)
            prev_pos = np.copy(self.cur_pos)

            if viz_3D_publisher is not None:
                # all_objects = [
                #     Object(
                #         pos=cur_pos, radius=Net2World * Params.agent_radius, ori=cur_ori_quat)
                # ]

                # viz_3D_publisher.publish(objects=all_objects, object_colors=[
                #     Params.agent_color_rgb, ])
                pass

            rospy.sleep(self.ros_delay)

        print("Final joint error: ", joint_error)

    def reach_start_pos(self, start_pose, goal_pose, object_poses, object_radii=None, goal_rot_radius=None,
                        pose_tol=0.03, dpose_tol=1e-2, reaching_dstep=0.1, clip_movement=False, viz_3D_publisher=None):

        start_pos = start_pose[:3]
        start_ori_quat = start_pose[3:]
        if self.debug:
            self.cur_pos = np.copy(start_pos)
            self.cur_ori_quat = np.copy(start_ori_quat)
            return

        else:
            start_dist = np.linalg.norm(start_pos - self.cur_pos)
            pose_error = 1e10
            dEE_pos = 1e10
            dEE_pos_running_avg = RunningAverage(length=5, init_vals=1e10)
            prev_pos = None
            while not rospy.is_shutdown() and (
                    self.cur_pos is None or (pose_error > pose_tol and dEE_pos_running_avg.avg > dpose_tol)):

                if self.cur_pos is None:
                    print("Waiting to receive robot pos")
                    rospy.sleep(self.ros_delay)
                    continue

                pos_vec = start_pos - self.cur_pos
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
                target_pos = self.cur_pos + pos_vec
                # target_pos = cur_pos + np.array([0, -0.1, 0.2])
                dist_to_start_ratio = min(pos_mag / (start_dist + 1e-5), 1.0)
                target_ori_quat = interpolate_rotations(
                    start_quat=self.cur_ori_quat, stop_quat=start_ori_quat,
                    alpha=1 - dist_to_start_ratio)
                self.pose_pub.publish(
                    pose_to_msg(np.concatenate([target_pos, target_ori_quat]), frame=ROBOT_FRAME))
                self.is_intervene_pub.publish(False)

                # Visualize
                if viz_3D_publisher is not None:
                    assert object_radii is not None
                    assert goal_rot_radius is not None
                    object_colors = [
                        (0, 255, 0),
                    ]
                    all_object_colors = object_colors + [
                        Params.start_color_rgb,
                        Params.goal_color_rgb,
                        Params.agent_color_rgb,
                    ]
                    # force_colors = object_colors + [Params.goal_color_rgb]
                    all_objects = [Object(pos=pose[0:POS_DIM], ori=[0.7627784, -0.00479786, 0.6414479, 0.08179578],
                                          radius=radius) for pose, radius in
                                   zip(object_poses, object_radii)]
                    all_objects += [
                        Object(
                            pos=start_pose[0:POS_DIM], radius=Params.agent_radius, ori=start_pose[POS_DIM:]),
                        Object(
                            pos=goal_pose[0:POS_DIM], radius=goal_rot_radius.item(), ori=goal_pose[POS_DIM:]),
                        Object(
                            pos=self.cur_pos, radius=Params.agent_radius, ori=self.cur_ori_quat)
                    ]
                    viz_3D_publisher.publish(
                        objects=all_objects, object_colors=all_object_colors,)

                cur_pose = np.concatenate([self.cur_pos, self.cur_ori_quat])
                pose_error = calc_pose_error(
                    cur_pose, start_pose, rot_scale=0.1)
                if prev_pos is not None:
                    dEE_pos = np.linalg.norm(self.cur_pos - prev_pos)
                dEE_pos_running_avg.update(dEE_pos)
                prev_pos = np.copy(self.cur_pos)
                print("Waiting to reach start pos (%s), cur pos (%s) error: %.3f,  change: %.3f" %
                      (np.array2string(target_pos, precision=2), np.array2string(self.cur_pos, precision=2),
                       pose_error, dEE_pos_running_avg.avg))
                rospy.sleep(self.ros_delay)

            rospy.sleep(0.5)  # pause to let arm finish converging

            print("Final error: ", pose_error)
            print(self.cur_pos, start_pos)
            print("Cur joints: ", self.cur_joints)

    def command_kinova_gripper(self, cmd_open):
        msg = Bool(cmd_open)
        for i in range(5):
            self.gripper_pub.publish(msg)
            rospy.sleep(0.1)

    def publish_is_intervene(self):
        self.is_intervene_pub.publish(self.is_intervene)

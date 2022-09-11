"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
from gc import collect
import numpy as np
import os
import argparse
import ipdb
import json
import typing
from tqdm import tqdm
import copy
# from pynput import keyboard
import re

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float32, Float64MultiArray

import torch
import sys

from globals import *
add_paths()  # sets the paths for the below imports

from model import Policy, PolicyNetwork, pose_to_model_input, decode_ori
from train import random_seed_adaptation, process_single_full_traj, DEVICE
from data_params import Params
from exp_params import *
from exp_utils import *
from elastic_band import Object
from kinova_interface import KinovaInterface
from viz_3D import Viz3DROSPublisher

import signal

signal.signal(signal.SIGINT, sigint_handler)

World2Net = 10.0
Net2World = 1 / World2Net

DEBUG = False
if DEBUG:
    dstep = 0.05
    ros_delay = 0.1
else:
    dstep = 0.12
    ros_delay = 0.4  # NOTE: if modify this, must also modify rolling avg window of dpose

inspection_radii = np.array([5.0])[:, np.newaxis]  # defined on net scale
inspection_rot_radii = np.array([4.0])[:, np.newaxis]
goal_rot_radius = np.array([4.0])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store',
                        type=str, help="trained model name", default="policy_3D")
    parser.add_argument('--loaded_epoch', action='store',
                        type=int, default=100)
    parser.add_argument('--view_ros', action='store_true',
                        help="visualize 3D scene with ROS Rviz")
    parser.add_argument('--user', action='store', type=str,
                        default="some_user", help="user name to save results")
    parser.add_argument('--trial', action='store', type=int, default=0)
    parser.add_argument('--run_eval', action='store_true')
    parser.add_argument('--collected_folder', action='store', type=str)
    args = parser.parse_args()

    return args


def run_adaptation(policy, kinova, collected_folder):
    files = os.listdir(collected_folder)
    exp_iter = None
    for f in files:
        matches = re.findall("perturb_traj_iter_(\d+)_num_\d+.npy", f)
        if len(matches) > 0:
            import ipdb
            ipdb.set_trace()
            exp_iter = int(matches[0])

    kinova.perturb_pose_traj = np.load(os.path.join(
        collected_folder, f"perturb_traj_iter_{exp_iter}_num_0.npy"))
    start_pos_world = start_poses[exp_iter]
    start_ori_quat = start_ori_quats[exp_iter]
    start_pose_world = np.concatenate([start_pos_world, start_ori_quat])
    start_pose_net = np.concatenate(
        [start_pos_world * World2Net, start_ori_quat])
    start_tensor = torch.from_numpy(
        pose_to_model_input(start_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)

    # Set goal robot pose
    goal_pos_world = goal_poses[exp_iter]
    goal_ori_quat = goal_ori_quats[exp_iter]
    goal_pose_net = np.concatenate(
        [goal_pos_world * World2Net, goal_ori_quat])
    goal_pose_world = np.concatenate([goal_pos_world, goal_ori_quat])

    inspection_pos_world = inspection_poses[exp_iter]
    inspection_ori_quat = inspection_ori_quats[exp_iter]
    inspection_pose_net = np.concatenate(
        [World2Net * inspection_pos_world, inspection_ori_quat], axis=-1)[np.newaxis]
    inspection_pose_tensor = torch.from_numpy(pose_to_model_input(
        inspection_pose_net)).to(torch.float32).to(DEVICE)

    object_poses_net = inspection_pose_net
    objects_tensor = inspection_pose_tensor
    objects_torch = torch.cat(
        [objects_tensor, object_radii_torch], dim=-1).unsqueeze(0)
    objects_rot_torch = torch.cat(
        [objects_tensor, object_rot_radii_torch], dim=-1).unsqueeze(0)

    assert len(
        kinova.perturb_pose_traj) > 1, "Need intervention traj of > 1 steps"
    dist = np.linalg.norm(
        kinova.perturb_pose_traj[-1][0:POS_DIM] - kinova.perturb_pose_traj[0][0:POS_DIM])
    # 1 step for start, 1 step for goal at least
    # T = max(2, int(np.ceil(dist / dstep)))
    T = 5
    perturb_pos_traj_world_interp = np.linspace(
        start=kinova.perturb_pose_traj[0][0:POS_DIM], stop=kinova.perturb_pose_traj[-1][0:POS_DIM], num=T)
    final_perturb_ori = kinova.perturb_pose_traj[-1][POS_DIM:]

    perturb_ori_traj = np.copy(final_perturb_ori)[
        np.newaxis, :].repeat(T, axis=0)
    perturb_pos_traj_net = perturb_pos_traj_world_interp * World2Net
    perturb_pose_traj_net = np.hstack(
        [np.vstack(perturb_pos_traj_net), perturb_ori_traj])
    sample = (perturb_pose_traj_net, start_pose_net, goal_pose_net, goal_rot_radius,
              object_poses_net[np.newaxis].repeat(T, axis=0),
              object_radii[np.newaxis].repeat(T, axis=0),
              object_idxs)
    processed_sample = process_single_full_traj(sample)

    # Update position
    print("Position:")
    dist_perturbed = np.linalg.norm(
        perturb_pos_traj_world_interp[0] - perturb_pos_traj_world_interp[-1])
    if dist_perturbed < 0.1:
        print(
            "No major position perturbation, skipping position adaptation...")
    else:
        best_pos_feats, _, _ = random_seed_adaptation(policy, processed_sample, train_pos=True, train_rot=False,
                                                      is_3D=True, num_objects=num_objects, loss_prop_tol=0.8,
                                                      pos_feat_max=pos_feat_max, pos_feat_min=pos_feat_min,
                                                      rot_feat_max=rot_feat_max, rot_feat_min=rot_feat_min,
                                                      pos_requires_grad=pos_requires_grad)

    # Update rotation
    print("Rotation:")
    _, best_rot_feats, best_rot_offsets = (
        random_seed_adaptation(policy, processed_sample, train_pos=False, train_rot=True,
                               is_3D=True, num_objects=num_objects, loss_prop_tol=0.2,
                               pos_feat_max=pos_feat_max, pos_feat_min=pos_feat_min,
                               rot_feat_max=rot_feat_max, rot_feat_min=rot_feat_min,
                               rot_requires_grad=rot_requires_grad))

    policy.update_obj_feats(best_pos_feats, best_rot_feats, best_rot_offsets)


if __name__ == "__main__":
    ########################################################
    args = parse_arguments()
    argparse_dict = vars(args)
    kinova = KinovaInterface(debug=DEBUG)

    # define save path
    trial_num = len(os.listdir("exp1/opa_saved_trials"))
    save_folder = f"exp1/opa_saved_trials/trial_{trial_num}"
    os.makedirs(save_folder)

    # Load trained model arguments
    model_root = "../../saved_model_files"
    model_name = "policy_3D"
    loaded_epoch = 100
    with open(os.path.join(model_root, model_name, "train_args_pt_1.json"), "r") as f:
        train_args = json.load(f)

    # load model
    is_3D = train_args["is_3D"]
    assert is_3D, "Run experiments with 3D models to control EE in 3D space"
    pos_dim, rot_dim = 3, 6
    network = PolicyNetwork(n_objects=Params.n_train_objects, pos_dim=pos_dim, rot_dim=rot_dim,
                            pos_preference_dim=train_args['pos_preference_dim'],
                            rot_preference_dim=train_args['rot_preference_dim'],
                            hidden_dim=train_args['hidden_dim'],
                            device=DEVICE).to(DEVICE)
    network.load_state_dict(
        torch.load(
            os.path.join(model_root, args.model_name,
                         "model_%d.h5" % loaded_epoch),
            map_location=DEVICE))
    policy = Policy(network)

    # define scene objects and whether or not we care about their pos/ori
    # In exp3, don't know anything about the scanner or table. Initialize both
    # position and orientation features as None with requires_grad=True
    calc_rot, calc_pos = True, True
    train_rot, train_pos = True, True
    # NOTE: not training "object_types", purely object identifiers
    object_idxs = np.arange(num_objects)
    pos_obj_types = [None]
    pos_requires_grad = [True]
    rot_obj_types = [None]
    rot_requires_grad = [True]

    rot_offsets_debug = None
    policy.init_new_objs(pos_obj_types=pos_obj_types, rot_obj_types=rot_obj_types,
                         pos_requires_grad=pos_requires_grad, rot_requires_grad=rot_requires_grad,
                         rot_offsets=rot_offsets_debug)
    obj_pos_feats = policy.obj_pos_feats

    # Convert np arrays to torch tensors for model input
    agent_radius_tensor = torch.tensor(
        [Params.agent_radius], device=DEVICE).view(1, 1)
    goal_rot_radii = torch.from_numpy(goal_rot_radius).to(
        torch.float32).to(DEVICE).view(1, 1)
    object_idxs_tensor = torch.from_numpy(object_idxs).to(
        torch.long).to(DEVICE).unsqueeze(0)
    inspection_radii_torch = torch.from_numpy(inspection_radii).to(
        torch.float32).to(DEVICE).view(-1, 1)
    inspection_rot_radii_torch = torch.from_numpy(inspection_rot_radii).to(
        torch.float32).to(DEVICE).view(-1, 1)
    object_radii_torch = inspection_radii_torch
    object_rot_radii_torch = inspection_rot_radii_torch
    object_radii = inspection_radii

    # Optionally view with ROS Rviz
    if args.view_ros:
        viz_3D_publisher = Viz3DROSPublisher(
            num_objects=num_objects, frame=ROBOT_FRAME)
        # TODO:
        object_colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
        ]
        all_object_colors = object_colors + [
            Params.start_color_rgb,
            Params.goal_color_rgb,
            Params.agent_color_rgb,
        ]
        force_colors = object_colors + [Params.goal_color_rgb]

    # Define pretrained feature bounds for random initialization
    pos_feat_min = torch.min(policy.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX],
                             policy.policy_network.pos_pref_feat_train[Params.REPEL_IDX]).item()
    pos_feat_max = torch.max(policy.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX],
                             policy.policy_network.pos_pref_feat_train[Params.REPEL_IDX]).item()

    rot_feat_min = torch.min(policy.policy_network.rot_pref_feat_train[Params.CARE_ROT_IDX],
                             policy.policy_network.rot_pref_feat_train[Params.IGNORE_ROT_IDX]).item()
    rot_feat_max = torch.max(policy.policy_network.rot_pref_feat_train[Params.CARE_ROT_IDX],
                             policy.policy_network.rot_pref_feat_train[Params.IGNORE_ROT_IDX]).item()
    pos_attract_feat = policy.policy_network.pos_pref_feat_train[Params.ATTRACT_IDX].detach(
    )

    if args.run_eval:
        assert args.collected_folder is not None
        run_adaptation(policy, kinova, collected_folder=args.collected_folder)

    it = 0
    pose_error_tol = 0.1
    max_pose_error_tol = 0.2  # too far to be considered converged and not moving
    del_pose_tol = 0.005  # over del_pose_interval iterations
    kinova.perturb_pose_traj = []
    override_pred_delay = False
    # [box1 intervene rot finish (don't reset to start), box2 watch,
    # box3 (pretend, need to buy) add new obstacles show online repulsion from them,
    # box4 watch final behavior
    # can1 intervene rot finish, can2 watch, can3 stand up and watch
    kinova.command_kinova_gripper(cmd_open=True)
    num_exps = len(start_poses)
    num_exps = 10
    for exp_iter in range(num_exps):
        # set extra mass of object to pick up
        # exp_iter = num_exps - 1
        # exp_iter = min(exp_iter, num_exps - 1)
        extra_mass = extra_masses[0]

        # Set start robot pose
        start_pos_world = start_poses[0]
        start_ori_quat = start_ori_quats[0]
        start_pose_world = np.concatenate([start_pos_world, start_ori_quat])
        start_pose_net = np.concatenate(
            [start_pos_world * World2Net, start_ori_quat])
        start_tensor = torch.from_numpy(
            pose_to_model_input(start_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)

        # Set goal robot pose
        goal_pos_world = goal_poses[0]
        goal_ori_quat = goal_ori_quats[0]
        goal_pose_net = np.concatenate(
            [goal_pos_world * World2Net, goal_ori_quat])
        goal_pose_world = np.concatenate([goal_pos_world, goal_ori_quat])

        inspection_pos_world = inspection_poses[0]
        inspection_ori_quat = inspection_ori_quats[0]
        inspection_pose_net = np.concatenate(
            [World2Net * inspection_pos_world, inspection_ori_quat], axis=-1)[np.newaxis]
        inspection_pose_tensor = torch.from_numpy(pose_to_model_input(
            inspection_pose_net)).to(torch.float32).to(DEVICE)

        object_poses_net = inspection_pose_net
        objects_tensor = inspection_pose_tensor
        objects_torch = torch.cat(
            [objects_tensor, object_radii_torch], dim=-1).unsqueeze(0)
        objects_rot_torch = torch.cat(
            [objects_tensor, object_rot_radii_torch], dim=-1).unsqueeze(0)

        if not DEBUG:
            kinova.reach_joints(HOME_JOINTS)

        perform_grasp(start_pose_world, item_ids[0], kinova)

        # initialize target pose variables
        local_target_pos_world = np.copy(kinova.cur_pos)
        local_target_ori = np.copy(kinova.cur_ori_quat)

        intervene_count = 0
        pose_error = 1e10
        del_pose = 1e10
        del_pose_running_avg = RunningAverage(length=5, init_vals=1e10)

        ee_pose_traj = []
        is_intervene_traj = []

        prev_pose_world = None
        while (not rospy.is_shutdown() and pose_error > pose_error_tol and
                (pose_error > max_pose_error_tol)):
            cur_pose_world = np.concatenate(
                [kinova.cur_pos.copy(), kinova.cur_ori_quat.copy()])
            cur_pos_net = kinova.cur_pos * World2Net
            cur_pose_net = np.concatenate([cur_pos_net, kinova.cur_ori_quat])
            pose_error = calc_pose_error(
                goal_pose_world, cur_pose_world, rot_scale=0)
            if prev_pose_world is not None:
                del_pose = calc_pose_error(prev_pose_world, cur_pose_world)
                del_pose_running_avg.update(del_pose)

            ee_pose_traj.append(cur_pose_world.copy())
            is_intervene_traj.append(kinova.is_intervene)
            print("CUR POSE:")
            print(ee_pose_traj[-1])

            if kinova.need_update and not DEBUG:
                # Hold current pose while running adaptation
                for i in range(5):
                    kinova.pose_pub.publish(pose_to_msg(
                        cur_pose_world, frame=ROBOT_FRAME))
                rospy.sleep(0.1)
                kinova.is_intervene = False
                kinova.publish_is_intervene()

                assert len(
                    kinova.perturb_pose_traj) > 1, "Need intervention traj of > 1 steps"
                dist = np.linalg.norm(
                    kinova.perturb_pose_traj[-1][0:POS_DIM] - kinova.perturb_pose_traj[0][0:POS_DIM])
                # 1 step for start, 1 step for goal at least
                # T = max(2, int(np.ceil(dist / dstep)))
                T = 5
                perturb_pos_traj_world_interp = np.linspace(
                    start=kinova.perturb_pose_traj[0][0:POS_DIM], stop=kinova.perturb_pose_traj[-1][0:POS_DIM], num=T)
                final_perturb_ori = kinova.perturb_pose_traj[-1][POS_DIM:]

                perturb_ori_traj = np.copy(final_perturb_ori)[
                    np.newaxis, :].repeat(T, axis=0)
                perturb_pos_traj_net = perturb_pos_traj_world_interp * World2Net
                perturb_pose_traj_net = np.hstack(
                    [np.vstack(perturb_pos_traj_net), perturb_ori_traj])
                sample = (perturb_pose_traj_net, start_pose_net, goal_pose_net, goal_rot_radius,
                          object_poses_net[np.newaxis].repeat(T, axis=0),
                          object_radii[np.newaxis].repeat(T, axis=0),
                          object_idxs)
                processed_sample = process_single_full_traj(sample)

                # For analysis, save weights before update
                torch.save(
                    {
                        "obj_pos_feats": policy.obj_pos_feats,
                        "obj_rot_feats": policy.obj_rot_feats,
                        "obj_rot_offsets": policy.obj_rot_offsets
                    },
                    f"{save_folder}/pre_adaptation_saved_weights_iter_{exp_iter}_num_{intervene_count}.pth"
                )

                # Update position
                print("Position:")
                dist_perturbed = np.linalg.norm(
                    perturb_pos_traj_world_interp[0] - perturb_pos_traj_world_interp[-1])
                if dist_perturbed < 0.1:
                    print(
                        "No major position perturbation, skipping position adaptation...")
                else:
                    best_pos_feats, _, _ = random_seed_adaptation(policy, processed_sample, train_pos=True, train_rot=False,
                                                                  is_3D=True, num_objects=num_objects, loss_prop_tol=0.8,
                                                                  pos_feat_max=pos_feat_max, pos_feat_min=pos_feat_min,
                                                                  rot_feat_max=rot_feat_max, rot_feat_min=rot_feat_min,
                                                                  pos_requires_grad=pos_requires_grad)

                # Update rotation
                print("Rotation:")
                _, best_rot_feats, best_rot_offsets = (
                    random_seed_adaptation(policy, processed_sample, train_pos=False, train_rot=True,
                                           is_3D=True, num_objects=num_objects, loss_prop_tol=0.2,
                                           pos_feat_max=pos_feat_max, pos_feat_min=pos_feat_min,
                                           rot_feat_max=rot_feat_max, rot_feat_min=rot_feat_min,
                                           rot_requires_grad=rot_requires_grad))

                policy.update_obj_feats(
                    best_pos_feats, best_rot_feats, best_rot_offsets)

                torch.save(
                    {
                        "obj_pos_feats": best_pos_feats,
                        "obj_rot_feats": best_rot_feats,
                        "obj_rot_offsets": best_rot_offsets
                    },
                    f"{save_folder}/post_adaptation_saved_weights_iter_{exp_iter}_num_{intervene_count}.pth"
                )
                np.save(f"{save_folder}/perturb_traj_iter_{exp_iter}_num_{intervene_count}",
                        kinova.perturb_pose_traj)

                # reset the intervention data
                kinova.perturb_pose_traj = []
                kinova.need_update = False
                override_pred_delay = True

                # increment intervention count to avoid overwriting this intervention's data
                intervene_count += 1

                # reach back to the pose before p
                continue

                # break  # start new trial with updated weights

            elif it % 2 == 0 or override_pred_delay:
                print("new target", it)
                with torch.no_grad():
                    # Define "object" inputs into policy
                    # current
                    cur_pose_tensor = torch.from_numpy(
                        pose_to_model_input(cur_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
                    current = torch.cat(
                        [cur_pose_tensor, agent_radius_tensor], dim=-1).unsqueeze(1)
                    # goal
                    goal_tensor = torch.from_numpy(
                        pose_to_model_input(goal_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)
                    goal_radii = goal_radius_scale * torch.norm(
                        goal_tensor[:, :pos_dim] -
                        cur_pose_tensor[:, :pos_dim],
                        dim=-1).unsqueeze(0)
                    goal_rot_objects = torch.cat(
                        [goal_tensor, goal_rot_radii], dim=-1).unsqueeze(1)
                    goal_objects = torch.cat(
                        [goal_tensor, goal_radii], dim=-1).unsqueeze(1)
                    # start
                    start_rot_radii = torch.norm(start_tensor[:, :pos_dim] - cur_pose_tensor[:, :pos_dim],
                                                 dim=-1).unsqueeze(0)
                    start_rot_objects = torch.cat(
                        [start_tensor, start_rot_radii], dim=-1).unsqueeze(1)

                    # Get policy output, form into action
                    pred_vec, pred_ori, object_forces = policy(current=current,
                                                               start=start_rot_objects,
                                                               goal=goal_objects, goal_rot=goal_rot_objects,
                                                               objects=objects_torch,
                                                               objects_rot=objects_rot_torch,
                                                               object_indices=object_idxs_tensor,
                                                               calc_rot=calc_rot,
                                                               calc_pos=calc_pos)
                    local_target_pos_world = cur_pose_tensor[0,
                                                             0:pos_dim] * Net2World
                    local_target_pos_world = local_target_pos_world + \
                        dstep * pred_vec[0, :pos_dim]
                    local_target_pos_world = local_target_pos_world.detach().cpu().numpy()
                    local_target_ori = decode_ori(
                        pred_ori.detach().cpu().numpy()).flatten()

                    if exp_iter == 4 and override_pred_delay:
                        vec = np.array([-0.05, -0.1, 0.05])
                        vec = vec / np.linalg.norm(vec)
                        local_target_pos_world = kinova.cur_pos + dstep * vec

                override_pred_delay = False

            # Optionally view with ROS
            if args.view_ros:
                all_objects = [Object(pos=Net2World * pose[0:POS_DIM], ori=pose[POS_DIM:],
                                      radius=Net2World * radius) for pose, radius in
                               zip(object_poses_net, object_radii)]
                all_objects += [
                    Object(
                        pos=Net2World * start_pose_net[0:POS_DIM], radius=Net2World * Params.agent_radius, ori=start_pose_net[POS_DIM:]),
                    Object(
                        pos=Net2World * goal_pose_net[0:POS_DIM], radius=Net2World * goal_rot_radius.item(), ori=goal_pose_net[POS_DIM:]),
                    Object(
                        pos=Net2World * cur_pose_net[0:POS_DIM], radius=Net2World * Params.agent_radius, ori=cur_pose_net[POS_DIM:])
                ]
                agent_traj = np.vstack(
                    [Net2World * cur_pose_net, np.concatenate([Net2World * local_target_pos_world, kinova.cur_ori_quat])])
                if isinstance(object_forces, torch.Tensor):
                    object_forces = object_forces[0].detach().cpu().numpy()

                viz_3D_publisher.publish(objects=all_objects, agent_traj=agent_traj,
                                         expert_traj=None, object_colors=all_object_colors,
                                         object_forces=object_forces, force_colors_rgb=force_colors)

            # Apply low-pass filter to smooth out policy's sudden changes in orientation
            interp_rot = interpolate_rotations(
                start_quat=kinova.cur_ori_quat, stop_quat=local_target_ori, alpha=0.7)

            # Clip target EE position to bounds
            local_target_pos_world = np.clip(
                local_target_pos_world, a_min=kinova.ee_min_pos, a_max=kinova.ee_max_pos)

            # Publish is_intervene
            kinova.publish_is_intervene()

            # Publish target pose
            if not DEBUG:
                target_pose = np.concatenate(
                    [local_target_pos_world, interp_rot])
                kinova.pose_pub.publish(pose_to_msg(
                    target_pose, frame=ROBOT_FRAME))

            if DEBUG:
                kinova.cur_pos = local_target_pos_world
                kinova.cur_ori_quat = interp_rot

            if it % 2 == 0:
                print("Pos error: ", np.linalg.norm(
                    local_target_pos_world - kinova.cur_pos))
                print("Ori error: ", np.linalg.norm(
                    np.arccos(np.abs(kinova.cur_ori_quat @ local_target_ori))))
                print("Dpose: ", del_pose_running_avg.avg)
                print()

            it += 1
            prev_pose_world = np.copy(cur_pose_world)
            rospy.sleep(0.3)

        kinova.reach_pose(goal_pose_world)

        # Save robot traj and intervene traj
        print("FULL TRAJ:")
        print(ee_pose_traj)
        np.save(f"{save_folder}/ee_pose_traj_iter_{exp_iter}.npy", ee_pose_traj)
        np.save(f"{save_folder}/is_intervene_traj{exp_iter}.npy",
                is_intervene_traj)

        print(
            f"Finished! Error {pose_error} vs tol {pose_error_tol}, \nderror {del_pose_running_avg.avg} vs tol {del_pose_tol}")
        print("Opening gripper to release item")

        rospy.sleep(0.1)

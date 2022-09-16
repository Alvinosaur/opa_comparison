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
import time

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

import signal

signal.signal(signal.SIGINT, sigint_handler)

World2Net = 10.0
Net2World = 1 / World2Net

DEBUG = True
dstep = 0.05
ros_delay = 0.1

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

inspection_radii = np.array([6.0])[:, np.newaxis]  # defined on net scale
inspection_rot_radii = np.array([6.0])[:, np.newaxis]
goal_rot_radius = np.array([4.0])


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', action='store',
                        type=str, help="trained model name", default="policy_3D")
    parser.add_argument('--loaded_epoch', action='store',
                        type=int, default=100)
    parser.add_argument('--num_perturbs', action='store',
                        type=int, required=True)
    parser.add_argument('--view_ros', action='store_true',
                        help="visualize 3D scene with ROS Rviz")
    parser.add_argument('--max_adaptation_time_sec',
                        action='store', type=float)
    parser.add_argument('--collected_folder', action='store', type=str)
    args = parser.parse_args()

    return args


def run_adaptation(policy, collected_folder, clip_params, num_perturbs, max_adaptation_time_sec, save_folder):
    files = os.listdir(collected_folder)
    all_perturb_pose_traj = []
    for f in files:
        
        matches = re.findall("perturb_traj_iter_(\d+)_num_\d+.npy", f)
        if len(matches) > 0:

            if len(all_perturb_pose_traj) < num_perturbs:
                print(f)
                perturb_pose_traj = np.load(os.path.join(
                    collected_folder, f))
                all_perturb_pose_traj.append(perturb_pose_traj)
            else:
                break
    print("USING {} perturb trajs".format(len(all_perturb_pose_traj)))

    exp_iter = 0  # always the 0th iter setting is perturbation
    start_pos_world = start_poses[exp_iter]
    start_ori_quat = start_ori_quats[exp_iter]
    start_pose_net = np.concatenate(
        [start_pos_world * World2Net, start_ori_quat])

    # Set goal robot pose
    goal_pos_world = goal_poses[exp_iter]
    goal_ori_quat = goal_ori_quats[exp_iter]
    goal_pose_net = np.concatenate(
        [goal_pos_world * World2Net, goal_ori_quat])

    inspection_pos_world = inspection_poses[exp_iter]
    inspection_ori_quat = inspection_ori_quats[exp_iter]
    inspection_pose_net = np.concatenate(
        [World2Net * inspection_pos_world, inspection_ori_quat], axis=-1)[np.newaxis]

    object_poses_net = inspection_pose_net

    all_processed_samples = []
    for perturb_pose_traj in all_perturb_pose_traj:
        assert len(
            perturb_pose_traj) > 1, "Need intervention traj of > 1 steps"
        T = 5
        perturb_pos_traj_world_interp = np.linspace(
            start=perturb_pose_traj[0][0:POS_DIM], stop=perturb_pose_traj[-1][0:POS_DIM], num=T)
        final_perturb_ori = perturb_pose_traj[-1][POS_DIM:]

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
        all_processed_samples.append(processed_sample)

    start_time = time.time()
    if max_adaptation_time_sec is not None:
        max_adaptation_time_sec -= (time.time() - start_time)
    else:
        max_adaptation_time_sec = 1e10  # will still stop after max iters

    # allocate 1/2 time for pos adapt and rot adapt
    pos_adapt_time = max_adaptation_time_sec / 2
    rot_adapt_time = max_adaptation_time_sec / 2

    # Update position
    print("Position:")
    dist_perturbed = np.linalg.norm(
        perturb_pos_traj_world_interp[0] - perturb_pos_traj_world_interp[-1])
    if dist_perturbed < 0.1:
        print(
            "No major position perturbation, skipping position adaptation...")
    else:
        best_pos_feats, _, _ = random_seed_adaptation(policy, all_processed_samples, train_pos=True, train_rot=False,
                                                      is_3D=True, num_objects=num_objects, loss_prop_tol=1.0,
                                                      pos_feat_max=pos_feat_max, pos_feat_min=pos_feat_min,
                                                      rot_feat_max=rot_feat_max, rot_feat_min=rot_feat_min,
                                                      pos_requires_grad=pos_requires_grad,
                                                      clip_params=clip_params,
                                                      allowed_time=pos_adapt_time)

    # Update rotation
    print("Rotation:")
    _, best_rot_feats, best_rot_offsets = (
        random_seed_adaptation(policy, all_processed_samples, train_pos=False, train_rot=True,
                               is_3D=True, num_objects=num_objects, loss_prop_tol=0.2,
                               pos_feat_max=pos_feat_max, pos_feat_min=pos_feat_min,
                               rot_feat_max=rot_feat_max, rot_feat_min=rot_feat_min,
                               rot_requires_grad=rot_requires_grad,
                               allowed_time=rot_adapt_time))

    policy.update_obj_feats(best_pos_feats, best_rot_feats, best_rot_offsets)
    policy.save(os.path.join(save_folder, "opa_model.pth"))


if __name__ == "__main__":
    ########################################################
    args = parse_arguments()
    argparse_dict = vars(args)

    # define save path
    save_folder = f"exp1/opa_saved_trials_inspection/eval_perturbs_{args.num_perturbs}_time_{args.max_adaptation_time_sec}"
    os.makedirs(save_folder, exist_ok=True)

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

    assert args.collected_folder is not None
    CLIP_PARAMS = False
    run_adaptation(policy, collected_folder=args.collected_folder,
                   clip_params=CLIP_PARAMS, num_perturbs=args.num_perturbs,
                   max_adaptation_time_sec=args.max_adaptation_time_sec,
                   save_folder=save_folder)

    it = 0
    pose_error_tol = 0.1
    max_pose_error_tol = 0.2  # too far to be considered converged and not moving
    del_pose_tol = 0.005  # over del_pose_interval iterations

    num_exps = len(start_poses)
    # num_exps = 3

    def rand_pos_noise():
        return np.random.normal(loc=0, scale=0.05, size=3)
    def rand_rot_euler_noise():
        return np.random.normal(loc=0, scale=5 * np.pi / 180, size=3)

    for exp_iter in range(num_exps):
        print("NEW EXP ", exp_iter)

        for rand_trial in range(5):
            # Set start robot pose
            start_pos_world = start_poses[exp_iter] + rand_pos_noise()
            start_ori_quat = R.from_euler("XYZ",
                    R.from_quat(start_ori_quats[exp_iter]).as_euler("XYZ") + rand_rot_euler_noise()
            ).as_quat()
            start_pose_world = np.concatenate([start_pos_world, start_ori_quat])
            start_pose_net = np.concatenate(
                [start_pos_world * World2Net, start_ori_quat])
            start_tensor = torch.from_numpy(
                pose_to_model_input(start_pose_net[np.newaxis])).to(torch.float32).to(DEVICE)

            # Set goal robot pose
            goal_pos_world = goal_poses[exp_iter] + rand_pos_noise()
            goal_ori_quat = R.from_euler("XYZ",
                    R.from_quat(goal_ori_quats[exp_iter]).as_euler("XYZ") + rand_rot_euler_noise()
            ).as_quat()
            goal_pose_net = np.concatenate(
                [goal_pos_world * World2Net, goal_ori_quat])
            goal_pose_world = np.concatenate([goal_pos_world, goal_ori_quat])

            # Set inspection pose
            inspection_pos_world = inspection_poses[exp_iter] + rand_pos_noise()
            inspection_ori_quat = R.from_euler("XYZ",
                    R.from_quat(inspection_ori_quats[exp_iter]).as_euler("XYZ") + rand_rot_euler_noise()
            ).as_quat()
            inspection_pose_world = np.concatenate([inspection_pos_world, inspection_ori_quat])
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

            # initialize target pose variables
            cur_pos = np.copy(start_pose_world[0:3])
            cur_ori_quat = np.copy(start_pose_world[3:])

            intervene_count = 0
            pose_error = 1e10
            del_pose = 1e10
            del_pose_running_avg = RunningAverage(length=5, init_vals=1e10)

            ee_pose_traj = []

            prev_pose_world = None
            step = 0
            max_steps = 50
            while (not rospy.is_shutdown() and pose_error > pose_error_tol and
                    (pose_error > max_pose_error_tol) and step < max_steps):
                step += 1
                cur_pose_world = np.concatenate(
                    [cur_pos.copy(), cur_ori_quat.copy()])
                cur_pos_net = cur_pos * World2Net
                cur_pose_net = np.concatenate([cur_pos_net, cur_ori_quat])
                pose_error = calc_pose_error(
                    goal_pose_world, cur_pose_world, rot_scale=0)
                if prev_pose_world is not None:
                    del_pose = calc_pose_error(prev_pose_world, cur_pose_world)
                    del_pose_running_avg.update(del_pose)

                ee_pose_traj.append(cur_pose_world.copy())
                if step % 5 == 0:
                    print("dist_to_goal: ", pose_error)

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
                    goal_radii = goal_radius_scale_custom * torch.norm(
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

                # override_pred_delay = False
                cur_pos = local_target_pos_world
                cur_ori_quat = local_target_ori

                # Apply low-pass filter to smooth out policy's sudden changes in orientation
                # interp_rot = interpolate_rotations(
                #     start_quat=cur_ori_quat, stop_quat=local_target_ori, alpha=0.7)

                # pos_std = 0.01
                # rot_euler_std = 2 * np.pi / 180
                # pos_noise = np.random.normal(loc=0, scale=pos_std, size=3)
                # rot_noise = np.random.normal(
                #     loc=0, scale=rot_euler_std, size=3)
                # cur_pos = local_target_pos_world + pos_noise
                # cur_ori_quat = R.from_euler("XYZ",
                #                             R.from_quat(interp_rot).as_euler(
                #                                 "XYZ") + rot_noise
                #                             ).as_quat()
                # it += 1
                prev_pose_world = np.copy(cur_pose_world)
                # rospy.sleep(0.3)

            # Save robot traj and intervene traj
            print("Done!")
            np.savez(f"{save_folder}/ee_pose_traj_iter_{exp_iter}_rand_trial_{rand_trial}.npz", traj=ee_pose_traj, start_pose=start_pose_world, goal_pose=goal_pose_world, inspection_pose=inspection_pose_world)


"""
Run in opa_comparison/src folder (NOT in individual exp folder):

python exp1/opa_eval.py --collected_folder exp1/opa_saved_trials_inspection/perturb_collection
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os

from globals import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_folder', action='store',
                        type=str, help="saved trial trajectory folder")
    parser.add_argument('--perturb_folder', action='store',
                        type=str, help="perturbation trajectory folder")
    args = parser.parse_args()

    return args


def calc_regret(traj, obstacle_pose, desired_rot_offset):
    pos_cost = np.min(np.linalg.norm(
        traj[:, 0:3] - obstacle_pose[np.newaxis, 0:3], axis=-1))
    return pos_cost, 0


def pose_as_quat(pose):
    if pose.shape[-1] == 7:
        return pose
    elif pose.shape[-1] == 6:
        if len(pose.shape) > 1:
            return np.hstack([
                pose[:, 0:3],
                R.from_euler("XYZ", pose[:, 3:]).as_quat()
            ])
        else:
            return np.hstack([
                pose[0:3],
                R.from_euler("XYZ", pose[3:]).as_quat()
            ])
    else:
        raise Exception("Unexpected pose shape")


"""
Commands:
OPA:
python eval_saved_traj.py --trials_folder opa_saved_trials_inspection/eval --perturb_folder opa_saved_trials_inspection/perturb_collection

Unified:
python eval_saved_traj.py --trials_folder unified_saved_trials_inspection/eval --perturb_folder unified_saved_trials_inspection/perturb_collection

"""

if __name__ == "__main__":
    args = parse_arguments()
    num_exps = len(start_poses)

    # ! NOTE: always apply perturb at 0th iter for 0th human pose config
    perturb_iter = 0
    perturb_traj = np.load(os.path.join(
        args.perturb_folder, f"perturb_traj_iter_{perturb_iter}_num_0.npy"))

    all_pos_costs = []
    all_rot_costs = []
    total_cost = 0.0
    for exp_iter in range(num_exps):
        pos_costs_iter = []
        rot_costs_iter = []
        for rand_trial in range(10):
            ee_pose_traj_data = np.load(os.path.join(
                args.trials_folder, f"ee_pose_traj_iter_{exp_iter}_rand_trial_{rand_trial}.npz"), allow_pickle=True)
            ee_pose_traj = pose_as_quat(ee_pose_traj_data["traj"])
            obstacle_pose = pose_as_quat(np.hstack([ee_pose_traj_data["obstacle_pose"][:3], np.array([0, 0, 0, 1])]))

            pos_cost, rot_cost = calc_regret(ee_pose_traj, obstacle_pose=obstacle_pose,
                            desired_rot_offset=None)

            # each row is all the data from one exp_iter
            pos_costs_iter.append(pos_cost)
            rot_costs_iter.append(rot_cost)

        all_pos_costs.append(pos_costs_iter)
        all_rot_costs.append(rot_costs_iter)

    all_pos_costs = np.array(all_pos_costs)
    all_rot_costs = np.array(all_rot_costs)
    np.savez(os.path.join(args.trials_folder, "metrics.npz"),
            all_pos_costs=all_pos_costs,
            all_rot_costs=all_rot_costs)
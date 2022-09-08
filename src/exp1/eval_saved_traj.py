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


def calc_regret(traj, human_pose, desired_rot_offset):
    # NOTE: right-multiply rot offset to get relative to human pose
    desired_rot = (R.from_quat(human_pose[3:]) *
                   R.from_quat(desired_rot_offset)).as_quat()
    regret = np.min(np.linalg.norm(
        traj[:, 0:3] - human_pose[np.newaxis, 0:3], axis=-1))

    # (T x 4) * (4,) = (T,)
    regret += np.min(
        np.arccos(np.abs(traj[:, 3:] @ desired_rot))
    )

    return regret


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
    inspection_ori_quat_from_perturb = inspection_ori_quats[perturb_iter]

    # Right-multiply rotational offset with inspection to get relative to inspector pose
    # R_perturb = R_inspection(FROM THE ORIGINAL PERTURBATION) * R_desired_offset
    # -> (left-mult) inv(R_inspection) * R_perturb = R_desired_offset
    # You can verify this makes sense by plotting with plot_ee_traj.py
    # which shows the estimatedd desired orientation
    desired_rot_offset = (
        R.from_quat(inspection_ori_quat_from_perturb).inv() *
        R.from_quat(perturb_traj[-1, 3:])).as_quat()

    total_cost = 0.0
    for exp_iter in range(num_exps):
        inspection_pos_world = inspection_poses[exp_iter]
        inspection_ori_quat = inspection_ori_quats[exp_iter]
        inspection_pose_net = np.concatenate(
            [inspection_pos_world, inspection_ori_quat], axis=-1)

        ee_pose_traj = np.load(os.path.join(
            args.trials_folder, f"ee_pose_traj_iter_{exp_iter}.npy"))

        cost = calc_regret(ee_pose_traj, human_pose=inspection_pose_net,
                           desired_rot_offset=desired_rot_offset)

        print(cost)

        total_cost += cost

    print("Total cost: ", total_cost)

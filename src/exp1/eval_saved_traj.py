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
    import ipdb
    ipdb.set_trace()
    pos_cost = np.mean(np.linalg.norm(
        traj[:, 0:3] - human_pose[np.newaxis, 0:3], axis=-1))

    # (T x 4) * (4,) = (T,)
    # TODO: try both min and mean
    rot_cost = np.mean(
        np.arccos(np.abs(traj[:, 3:] @ desired_rot))
    )

    return pos_cost, rot_cost


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
python eval_saved_traj.py --trials_folder opa_saved_trials_obstacle1/eval_perturbs_1_time_1.0 --perturb_folder opa_saved_trials_obstacle1/perturb_collection

Unified:
python eval_saved_traj.py --trials_folder unified_saved_trials_inspection/eval --perturb_folder unified_saved_trials_inspection/perturb_collection

"""

# folders = os.("/home/ruic/Documents/opa/opa_comparison/src/exp1")

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
            inspection_pose = pose_as_quat(ee_pose_traj_data["inspection_pose"])

            pos_cost, rot_cost = calc_regret(ee_pose_traj, human_pose=inspection_pose,
                            desired_rot_offset=desired_rot_offset)

            import ipdb
            ipdb.set_trace()

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
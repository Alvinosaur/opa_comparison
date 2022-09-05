import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os

from globals import *

"""
exp1 involves ... Thus, we define regret in this scenario as ...
This file should be called on each of the final saved trajectories of FERL, OPA, Unified, and 2017 paper to quantiatively compare their performance.
"""

# TODO: something like this but without the if statements since we will just create separate folders for each experiment.
# trajectory reward function


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_folder', action='store',
                        type=str, help="saved trial trajectory folder")
    parser.add_argument('--perturb_folder', action='store',
                        type=str, help="perturbation trajectory folder")
    args = parser.parse_args()

    return args


def calc_regret(traj, human_pose, desired_rot_offset=DESIRED_ROT_OFFSET):
    # cost weights so that each cost has similar magnitude
    # TODO: tune this!
    dist_weight = 1.0
    rot_weight = 1.0

    # NOTE: right-multiply rot offset to get relative to human pose
    desired_rot = (R.from_quat(human_pose[3:]) *
                   R.from_quat(desired_rot_offset)).as_quat()
    # regret = 0
    # for i in range(len(traj)):
    #     # distance from human
    #     regret += dist_weight * np.linalg.norm(traj[i, 0:3] - human_pose[0:3])

    #     # error in orientation
    #     regret += rot_weight * np.arccos(np.abs(traj[i, 3:] @ desired_rot))

    regret = dist_weight * np.min(np.linalg.norm(
        traj[:, 0:3] - human_pose[np.newaxis, 0:3], axis=-1))

    import ipdb
    ipdb.set_trace()  # TODO: verify below and verify dist_weight/rot_weight
    regret += rot_weight * np.min(
        np.arccos(
            np.abs(
                np.einsum("ti,tj->t", traj[:, 3:], desired_rot[np.newaxis, :])
            )
        )
    )

    return regret


if __name__ == "__main__":
    args = parse_arguments()
    num_exps = len(start_poses)
    perturb_traj = np.load(os.path.join(
        args.perturb_folder, "perturb_traj_iter_1_num_0.npy"))
    desired_rot_offset = perturb_traj[-1, 3:]
    total_cost = 0.0
    for exp_iter in range(num_exps):
        inspection_pos_world = inspection_poses[exp_iter]
        inspection_ori_quat = inspection_ori_quats[exp_iter]
        inspection_pose_net = np.concatenate(
            [inspection_pos_world, inspection_ori_quat], axis=-1)

        ee_pose_traj = np.load(os.path.join(
            args.trials_folder, "ee_pose_traj_iter_0.npy"))

        cost = calc_regret(ee_pose_traj, human_pose=inspection_pose_net,
                           desired_rot_offset=desired_rot_offset)

        total_cost += cost

    print("Total cost: ", total_cost)

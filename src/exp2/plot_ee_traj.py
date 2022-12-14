from ast import parse
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import argparse
import re

from globals import *

from model import decode_ori


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_coordinate_frame(ax, T, R):
    tx, ty, tz = T
    scale = 0.15
    new_x = scale * R @ np.array([1, 0, 0.0]) + T
    new_y = scale * R @ np.array([0, 1, 0.0]) + T
    new_z = scale * R @ np.array([0, 0, 1.0]) + T

    arrow_prop_dict = dict(
        mutation_scale=20, arrowstyle='->', shrinkA=0, shrinkB=0)

    a = Arrow3D([tx, new_x[0]], [ty, new_x[1]], [
                tz, new_x[2]], **arrow_prop_dict, color='r')
    ax.add_artist(a)
    a = Arrow3D([tx, new_y[0]], [ty, new_y[1]], [
                tz, new_y[2]], **arrow_prop_dict, color='b')
    ax.add_artist(a)
    a = Arrow3D([tx, new_z[0]], [ty, new_z[1]], [
                tz, new_z[2]], **arrow_prop_dict, color='g')
    ax.add_artist(a)


def load_pose_as_quat(pose):
    if pose.shape[-1] == 7:
        return pose
    elif pose.shape[-1] == 6:
        print("CONVERTING EULER TO QUAT...")
        if len(pose.shape) > 1:
            return np.concatenate([pose[..., 0:3], R.from_euler("XYZ", pose[..., 3:]).as_quat()])
        else:
            return np.concatenate([pose[0:3], R.from_euler("XYZ", pose[3:]).as_quat()])
    else:
        raise Exception("Unexpected pose shape! ", pose.shape)

def view_trained_reward_traj(perturb_path, path):
    ax = plt.axes(projection='3d')

    perturb_iter = int(re.findall(
        "perturb_traj_iter_(\d+)_num_\d+.npy", perturb_path)[0])
    inspection_ori_quat_from_perturb = obstacle_ori_quats[perturb_iter]

    generated_traj_data = np.load(path, allow_pickle=True)
    ee_pose_traj = load_pose_as_quat(generated_traj_data["traj"])
    start_pose = load_pose_as_quat(generated_traj_data["start_pose"])
    goal_pose = load_pose_as_quat(generated_traj_data["goal_pose"])
    inspection_pose = load_pose_as_quat(np.hstack([generated_traj_data["obstacle_pose"][:3], np.array([0, 0, 0, 1])]))

    perturb_traj = np.load(perturb_path)
    ax.plot3D(perturb_traj[:, 0], perturb_traj[:, 1], perturb_traj[:, 2],
              label="perturb", color="black", linewidth=5)

    # draw estimated ground truth desired orientation
    # Right-multiply rotational offset with inspection to get relative to inspector pose
    # R_perturb = R_inspection(FROM THE ORIGINAL PERTURBATION) * R_desired_offset
    # -> (left-mult) inv(R_inspection) * R_perturb = R_desired_offset
    desired_rot_offset = (
        R.from_quat(inspection_ori_quat_from_perturb).inv() *
        R.from_quat(perturb_traj[-1, 3:])).as_quat()
    # NOTE: right-multiply rot offset to get relative to (FROM CURRENT SCENARIO) human pose
    desired_rot = (R.from_quat(inspection_pose[3:]) *
                   R.from_quat(desired_rot_offset)).as_quat()
    draw_coordinate_frame(ax,
                          T=perturb_traj[-1, 0:3],
                          R=R.from_quat(desired_rot).as_matrix())
    print("NOTE: Final Perturbation Orientation is estimated for the current human pose! Perturbation position traj isn't changed from the original recording!!!!!!!!")


    T = ee_pose_traj.shape[0]
    print(ee_pose_traj.shape[0])
    for t in range(ee_pose_traj.shape[0]):
        ax.plot3D(ee_pose_traj[t:t + 2, 0], ee_pose_traj[t:t + 2, 1],
                  ee_pose_traj[t:t + 2, 2], alpha=0.9, color=cm.jet(t / T), linewidth=3)

        if t % 3 == 0:
            draw_coordinate_frame(ax,
                                  T=ee_pose_traj[t, 0:3],
                                  R=R.from_quat(ee_pose_traj[t, 3:]).as_matrix())
            # draw_coordinate_frame(ax,
            #                       T=ee_pose_traj[t, 0:3],
            #                       R=R.from_quat(inspection_pose[3:]).as_matrix())

    ax.scatter(*start_pose[:3], label="Start")
    ax.scatter(*goal_pose[:3], label="Goal")
    ax.scatter(*inspection_pose[:3], label="Human")
    draw_coordinate_frame(ax,
                          T=inspection_pose[:3],
                          R=R.from_quat(inspection_pose[3:]).as_matrix())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()
    ax.clear()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', action='store', type=str, required=True)
    parser.add_argument('--perturb_path', action='store',
                        type=str, required=True)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_arguments()
    view_trained_reward_traj(path=args.path, perturb_path=args.perturb_path)


"""
Commands:
OPA executed traj 0 with perturb at iter 0:
    python plot_ee_traj.py --path opa_saved_trials_inspection/eval/ee_pose_traj_iter_0.npy --perturb_path opa_saved_trials_inspection/perturb_collection/perturb_traj_iter_0_num_0.npy

Unified:
    python3 plot_ee_traj.py --path online_is_expert_True_saved_trials_obstacle1//eval_perturbs_1_time_60.0/ee_pose_traj_iter_1_rand_trial_0.npz --perturb_path unified_saved_trials_obstacle1/perturb_collection/perturb_traj_iter_0_num_0.npy

Online:
    python3 plot_ee_traj.py --path online_is_expert_True_saved_trials_inspection/eval_perturbs_1_time_60.0/ee_pose_traj_iter_0_rand_trial_0.npz --perturb_path unified_saved_trials_obstacle1/perturb_collection/perturb_traj_iter_0_num_0.npy

    python3 plot_ee_traj.py --path online_is_expert_False_saved_trials_inspection/eval_perturbs_1_time_60.0/ee_pose_traj_iter_0_rand_trial_0.npz --perturb_path unified_saved_trials_obstacle1/perturb_collection/perturb_traj_iter_0_num_0.npy
"""

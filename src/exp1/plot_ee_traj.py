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


def view_trained_reward_traj(perturb_path, path):
    ax = plt.axes(projection='3d')

    exp_iter = int(re.findall("ee_pose_traj_iter_(\d+).npy", path)[0])

    perturb_traj = np.load(perturb_path)
    ax.plot3D(perturb_traj[:, 0], perturb_traj[:, 1], perturb_traj[:, 2],
              label="perturb", color="black", linewidth=5)
    draw_coordinate_frame(ax,
                          T=perturb_traj[-1, 0:3],
                          R=R.from_quat(perturb_traj[-1, 3:]).as_matrix())

    # Set start robot pose
    start_pos = start_poses[exp_iter]
    start_ori_quat = start_ori_quats[exp_iter]
    start_ori_euler = R.from_quat(start_ori_quat).as_euler("XYZ")
    start_pose = np.concatenate([start_pos, start_ori_euler])
    start_pose_quat = np.concatenate([start_pos, start_ori_quat])

    # Set goal robot pose
    goal_pos = goal_poses[exp_iter]
    goal_ori_quat = goal_ori_quats[exp_iter]
    goal_ori_euler = R.from_quat(goal_ori_quat).as_euler("XYZ")
    goal_pose = np.concatenate([goal_pos, goal_ori_euler])
    goal_pose_quat = np.concatenate([goal_pos, goal_ori_quat])

    # Set inspection pose
    inspection_pos = inspection_poses[exp_iter]
    inspection_ori_quat = inspection_ori_quats[exp_iter]
    inspection_ori_euler = R.from_quat(inspection_ori_quat).as_euler("XYZ")
    inspection_pose_euler = np.concatenate(
        [inspection_pos, inspection_ori_euler])

    ee_pose_traj = np.load(path)
    print(ee_pose_traj.shape[0])
    for t in range(ee_pose_traj.shape[0]):
        ax.plot3D(ee_pose_traj[t:t + 2, 0], ee_pose_traj[t:t + 2, 1],
                  ee_pose_traj[t:t + 2, 2], alpha=0.9, color=cm.jet(t / T), linewidth=3)

        if t % 3 == 0:
            draw_coordinate_frame(ax,
                                  T=ee_pose_traj[t, 0:3],
                                  R=R.from_quat(ee_pose_traj[t, 3:]).as_matrix())

    ax.scatter(*start_pose[:3], label="Start")
    ax.scatter(*goal_pose[:3], label="Goal")
    ax.scatter(*inspection_pos, label="Human")
    draw_coordinate_frame(ax,
                          T=inspection_pos,
                          R=R.from_quat(inspection_ori_quat).as_matrix())

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
OPA executed traj 0 with perturb at iter 1:
    python plot_ee_traj.py --path opa_saved_trials_inspection/eval/ee_pose_traj_iter_0.npy --perturb_path opa_saved_trials_inspection/perturb_collection/perturb_traj_iter_1_num_0.npy

Unified:
    python plot_ee_traj.py --path unified_saved_trials_inspection/eval/ee_pose_traj_iter_0.npy --perturb_path unified_saved_trials_inspection/perturb_collection/perturb_traj_iter_0_num_0.npy

"""

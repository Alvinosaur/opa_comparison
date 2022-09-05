from this import d
import numpy as np
import random
import argparse
import os
from scipy.spatial.transform import Rotation as R
import time
import signal
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from globals import *
add_paths()  # sets the paths for the below imports

# from exp_utils import *

from unified_trajopt import TrajOptBase
from unified_reward import TrainReward
from trajectory import Trajectory

seed = 123
random.seed(seed)
np.random.seed(seed)


def sigint_handler(signal, frame):
    # Force scripts to exit cleanly
    exit()


signal.signal(signal.SIGINT, sigint_handler)


"""
An extremely modified version of unified_main.py.
unified_main.py preserved the original logic of "test_main.py",
but just cleaned up bad coding practices that could lead
to bugs later.

NOTE: because image/video results not necessary, no need
for approach pose, grasp pose, grabbing object, etc..
just move EE

NOTE: we could make this cleaner by defining generic objects, 
generic start/goals that are read from an experiment-specific file,
but we don't have many experiments, so just copy all code to separate python
files.

NOTE: temporarily don't pick things up just to speed up testing,
then at the end we can add it back for final pictures if necessary

This file changes the logic heavily:
- no pre-loaded demo.pkl files
- no query/comparisons, only physical corrections
- pick up boxes
- allow for OPA state features to be used instead of raw object state 
for fair comparison with our approach since this could simplify learning
"""

# Hyperparameters
TRAJ_LEN = 10  # fixed number of wayponts for trajopt
waypts_time = np.linspace(0, T, TRAJ_LEN)
adapt_num_epochs = 5

DEVICE = "cpu"
print("DEVICE: %s" % DEVICE)

# Constants
reward_model1_path = "models/reward_model_1"
EXP_TYPE = "exp1"

DEBUG = False
if DEBUG:
    dstep = 0.05
    ros_delay = 0.1
else:
    dstep = 0.12
    ros_delay = 0.4  # NOTE: if modify this, must also modify rolling avg window of dpose


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
    scale = 0.2
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


class TrajOptExp(TrajOptBase):
    # experiment-specific oracle reward function used to generate "optimal"
    # human behavior
    POS_WEIGHT = 1.0
    ROT_WEIGHT = 0.75

    def __init__(self, human_pose_euler, context_dim, *args, **kwargs):
        super(TrajOptExp, self).__init__(*args, **kwargs)
        self.human_pose_euler = human_pose_euler

        # rotate relative to human facing exactly opposite of human
        # so desired orientation is 180 degrees rotated aboutu human's z-axis.
        self.desired_present_item_rot = (R.from_euler("xyz", human_pose_euler[3:7]) *
                                         R.from_euler("xyz", [0, 0, 180], degrees=True)).as_quat()

    # custom process context function
    def process_context(self, robot, context):
        human_pose_quat = context
        if not self.use_state_features:
            return human_pose_quat
        else:
            dist = np.linalg.norm(robot[0:3] - human_pose_quat[0:3])
            vec = (human_pose_quat[0:3] - robot[0:3]) / dist
            return np.concatenate([dist, vec, human_pose_quat[3:7]])

    def trajcost_true(self, xi):
        xi = xi.reshape(self.n_waypoints, self.state_dim)
        R = 0
        for idx in range(self.n_waypoints):
            # move closer to human
            reward_pos = - \
                np.linalg.norm(self.human_pose_quat[0:3] - xi[idx, 0:3])

            # present item rotated correctly
            reward_rot = -np.arccos(
                np.abs(self.desired_present_item_rot @ xi[idx, 3:7]))

            R += self.POS_WEIGHT * reward_pos + self.ROT_WEIGHT * reward_rot

        cost = -R
        return cost


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_state_features', action='store_true',
                        help="context are the same state features that OPA uses, not raw object state")
    parser.add_argument('--view_ros', action='store_true',
                        help="visualize 3D scene with ROS Rviz")
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    # Define reward model
    state_dim = 3 + 3  # (pos, rot euler)
    # context: human pose
    context_dim = 1 + 3 + 3 if args.use_state_features else 3 + 3
    input_dim = state_dim + context_dim  # robot pose, human pose
    rm1 = TrainReward(model_dim=(input_dim, 128),
                      epoch=500, traj_len=TRAJ_LEN, device=DEVICE)

    perturb_pose_traj = [None] * TRAJ_LEN
    num_exps = len(start_poses)
    for exp_iter in range(num_exps):
        # set extra mass of object to pick up
        extra_mass = extra_masses[exp_iter]

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

        # 1st iter: show default behavior
        # 2nd iter: perform adaptation on saved perturbation data
        # 3rd iter: visualize new behavior after adaptation
        if exp_iter == 1:
            perturb_pose_traj = np.load("/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials/trial_6/perturb_traj_iter_0_num_0.npy")

            # Downsample original traj
            perturb_pose_traj = Trajectory(
                waypts=perturb_pose_traj,
                waypts_time=np.linspace(0, 10, len(perturb_pose_traj))).downsample(num_waypts=40).waypts

            # convert quat to euler XYZ traj
            perturb_pose_traj = np.vstack(perturb_pose_traj)
            perturb_pose_traj_euler = np.hstack([
                perturb_pose_traj[:, 0:3],
                R.from_quat(perturb_pose_traj[:, 3:]).as_euler("XYZ")
            ])

            # Perform adaptation and re-run trajopt
            T = perturb_pose_traj.shape[0]
            context = inspection_pose_euler[np.newaxis, :].repeat(T, axis=0)
            rm1.train_rewards([perturb_pose_traj_euler, ], context=context)
            rm1.save(folder="", name="test_unified_models")

        trajopt = TrajOptExp(home=start_pose,
                             goal=goal_pose,
                             human_pose_euler=inspection_pose_euler,
                             context_dim=context_dim,
                             use_state_features=args.use_state_features,
                             waypoints=TRAJ_LEN)
        traj = Trajectory(
            waypts=trajopt.optimize(
                context=inspection_pose_euler, reward_model=rm1),
            waypts_time=waypts_time)

        np.save(f"test_unified_traj_{exp_iter}.npy", traj.waypts)


def plot_downsampled_perturbation():
    perturb_pose_traj = np.load("unified_perturb_pose_traj.npy")
    ax = plt.axes(projection='3d')
    ax.plot3D(perturb_pose_traj[:, 0],
              perturb_pose_traj[:, 1], perturb_pose_traj[:, 2])
    # ax.plot_surface(xx, yy, zz, alpha=0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    perturb_pose_traj_downsampled = Trajectory(
        waypts=perturb_pose_traj,
        waypts_time=np.linspace(0, 10, len(perturb_pose_traj))).downsample(num_waypts=40).waypts
    ax.scatter(perturb_pose_traj_downsampled[:, 0],
               perturb_pose_traj_downsampled[:, 1], perturb_pose_traj_downsampled[:, 2])
    plt.show()


def view_deformations():
    orig = np.load("unified_perturb_pose_traj.npy")
    deformations = np.load("deformed.npy", allow_pickle=True)
    ax = plt.axes(projection='3d')
    ax.plot3D(orig[:, 0], orig[:, 1], orig[:, 2],
              label="orig", color="black", linewidth=5)
    for i in range(10):
        random_deform = random.choice(deformations)
        ax.plot3D(random_deform[:, 0], random_deform[:, 1],
                  random_deform[:, 2])
        ax.scatter(random_deform[0, 0], random_deform[0, 1],
                   random_deform[0, 2])

    plt.legend()
    plt.show()


def view_trained_reward_traj():
    """
    Insights gained:
        - without the goal constraint, traj would just end at the final point  of intervention because highest reward there. This means  that much of the traj just stays there, so need to limit traj len
        - only 3 networks, will definitely face issue of catastrophhic forgetting that we can briefly mention
    """
    use_state_features = False
    state_dim = 3 + 3  # (pos, rot euler)
    # context: human pose
    context_dim = 1 + 3 + 3 if use_state_features else 3 + 3
    input_dim = state_dim + context_dim  # robot pose, human pose
    rm1 = TrainReward(model_dim=(input_dim, 128),
                      epoch=500, traj_len=TRAJ_LEN, device=DEVICE)

    # Load trained reward
    # rm1.load(folder="/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials/trial_6/", name="exp_0_adapt_iter_0")

    ax = plt.axes(projection='3d')
    perturb_traj = np.load("/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials/trial_6/perturb_traj_iter_0_num_0.npy")

    exp_iter = 0

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

    trajopt = TrajOptExp(home=start_pose,
                         goal=goal_pose,
                         human_pose_euler=inspection_pose_euler,
                         context_dim=context_dim,
                         use_state_features=use_state_features,
                         waypoints=10)
    # traj = Trajectory(
    #     waypts=trajopt.optimize(
    #         context=inspection_pose_euler, reward_model=rm1),
    #     waypts_time=waypts_time)
    # traj_wpts = traj.waypts

    traj_wpts = np.load("/home/ruic/Documents/opa/opa_comparison/src/test_unified_traj_2.npy")


    
    ax.plot3D(perturb_traj[:, 0], perturb_traj[:, 1], perturb_traj[:, 2],
              label="orig", color="black", linewidth=5)

    T = traj_wpts.shape[0]
    for t in range(T):
        ax.plot3D(traj_wpts[t:t + 2, 0], traj_wpts[t:t + 2, 1],
                  traj_wpts[t:t + 2, 2], alpha=0.9, color=cm.jet(t / T), linewidth=3)

        # if t % 3 == 0:
        #     draw_coordinate_frame(ax,
        #                           T=traj_wpts[t, 0:3],
        #                           R=R.from_euler("XYZ", traj_wpts[t, 3:]).as_matrix())

    ax.scatter(*start_pose[:3], label="Start")
    ax.scatter(*goal_pose[:3], label="Goal")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()
    ax.clear()


if __name__ == "__main__":
    # plot_downsampled_perturbation()
    # view_deformations()
    view_trained_reward_traj()
    # main()

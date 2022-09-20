import numpy as np
import torch
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint
import argparse
import copy

from globals import *

################## @Bo: Parameters to tune ######################
"""
I tuned these parameters by first looking at 1. how closely the  new traj matches the original and 2. how smooth it is. If the new traj doesn't match at all (ie: just lin interpolates straight  to goal), it's a sign that the max abs translation param is too low because new traj can't physically stay that close to the original traj while still reaching the goal. So you would need to increase it.
If traj is good but noisy, try either increasing the smoothness weight or redducing the max abs trans.
"""
max_abs_translation_btwn_two_pts = 0.05
max_abs_rotation_quat_btwn_two_pts = 0.8  # this may need tuning, not easy  for me to tell with plot_ee_Traj
smoothness_weight = 0.3

class TrajOpt(object):
    def __init__(self, home, goal, orig_traj, max_iter=1000, eps=1e-3, ftol=1e-6):
        self.home = home
        self.goal = goal
        self.orig_traj = orig_traj
        self.n_waypoints = len(orig_traj)
        self.max_iter = max_iter
        self.eps = eps
        self.count = 0
        self.ftol = ftol

        self.state_dim = len(self.home)

        # initial seed trajectory as linear interp from start to goal
        self.xi0 = np.zeros((self.n_waypoints, self.state_dim))
        for idx in range(self.n_waypoints):
            self.xi0[idx, :] = self.home + idx / \
                (self.n_waypoints - 1.0) * (self.goal - self.home)
        self.xi0 = self.xi0.reshape(-1)

        # Define start constraint that the first waypoint is the home position
        # B: (states x states*waypoints), x: (states*waypoints) reshaped from (waypoints x state_dim)
        # in row-major order, so each waypoint is a row.
        # self.B[0,0], self.B[1,1], ... self.B[6,6] = 1, 1, ... 1 refer to the first waypoint
        # cols [0, ..., state_dim] are 0th waypoint, one row for each state
        self.B = np.zeros((self.state_dim, self.state_dim * self.n_waypoints))
        for idx in range(self.state_dim):
            self.B[idx, idx] = 1
        # lb <= Bx <= ub
        self.start_con = LinearConstraint(self.B, self.home, self.home)

        # define goal constraint enforcing convergence to goal
        self.G = np.zeros((self.state_dim, self.state_dim * self.n_waypoints))
        last_wpt_idx = len(self.xi0) - self.state_dim
        for idx in range(self.state_dim):
            self.G[idx, last_wpt_idx + idx] = 1
        self.goal_con = LinearConstraint(self.G, self.goal, self.goal)

        # constrain max change between waypoints
        self.pos_action_con = NonlinearConstraint(
            self.pos_action_con_func, 
            -max_abs_translation_btwn_two_pts, 
            max_abs_translation_btwn_two_pts)
        self.rot_action_con = NonlinearConstraint(
            self.rot_action_con_func, 
            -max_abs_rotation_quat_btwn_two_pts, 
            max_abs_rotation_quat_btwn_two_pts)

        self.constraints = {self.start_con, self.goal_con}
        self.constraints.add(self.pos_action_con)
        self.constraints.add(self.rot_action_con)

    def pos_action_con_func(self, xi):
        xi = xi.reshape(self.n_waypoints, self.state_dim)
        actions = xi[1:, :3] - xi[:-1, :3]
        return actions.reshape(-1)

    def rot_action_con_func(self, xi):
        xi = xi.reshape(self.n_waypoints, self.state_dim)
        actions = xi[1:, 3:6] - xi[:-1, 3:6]
        return actions.reshape(-1)

    # trajectory cost function
    def trajcost(self, xi):
        xi = xi.reshape(self.n_waypoints, self.state_dim)
        error = np.linalg.norm(xi[1:-1, :] - self.orig_traj[1:-1, :], axis=-1)
        smoothness = smoothness_weight * np.linalg.norm(xi[1:, :] - xi[0:-1, :], axis=-1).mean()
        return np.max(error) + smoothness

    # run the optimizer
    def optimize(self, method='SLSQP'):
        res = minimize(self.trajcost,
                       self.xi0, method=method, constraints=self.constraints,
                       options={'eps': self.eps, 'maxiter': self.max_iter,
                       'ftol': self.ftol},
                       )

        xi = res.x.reshape(self.n_waypoints, self.state_dim)
        return xi

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_traj_path', action='store',
                        type=str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_arguments()

    saved_traj_data = dict(np.load(args.saved_traj_path))
    saved_traj = saved_traj_data["traj"]
    goal_pose_quat = saved_traj_data["goal_pose"]
    start_pose_quat = saved_traj_data["start_pose"]

    new_traj = TrajOpt(home=start_pose_quat, goal=goal_pose_quat, orig_traj=saved_traj).optimize()

    saved_traj_data["traj"] = new_traj

    np.savez(args.saved_traj_path + "_CONSTRAINED.npz", **saved_traj_data)


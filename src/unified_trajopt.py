import numpy as np
import torch
import time
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint, NonlinearConstraint


class TrajOptBase(object):

    def __init__(self, home, goal, state_len, use_state_features, waypoints=10,
                max_iter=1000, eps=1e-3):
        self.n_waypoints = waypoints
        self.state_len = state_len
        self.home = home
        self.goal = goal
        self.use_state_features = use_state_features
        self.max_iter = max_iter
        self.eps = eps

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
            self.pos_action_con_func, -0.2, 0.2)
        self.rot_action_con = NonlinearConstraint(
            self.rot_action_con_func, -0.8, 0.8)

        self.constraints = {self.start_con, self.goal_con, self.pos_action_con, self.rot_action_con}

    def pos_action_con_func(self, xi):
        xi = xi.reshape(self.n_waypoints, self.state_dim)
        actions = xi[1:, :3] - xi[:-1, :3]
        return actions.reshape(-1)

    def rot_action_con_func(self, xi):
        xi = xi.reshape(self.n_waypoints, self.state_dim)
        actions = xi[1:, 3:6] - xi[:-1, 3:6]
        return actions.reshape(-1)

    def process_context(self, robot, context):
        # default no change, can be changed to extract state-based features
        return context

    # trajectory cost function
    def trajcost(self, reward_model, context, xi):
        xi = xi.reshape(self.n_waypoints, self.state_dim)
        states = []
        # target = joint2pose(self.context)
        for idx in range(self.n_waypoints):
            context = self.process_context(xi[idx, :], context)
            states.append(np.concatenate([xi[idx, :], context], axis=None))
        states = np.vstack(states)
        states = torch.FloatTensor(states)
        R = reward_model.reward(states)
        return -R

    # run the optimizer
    def optimize(self, reward_model, context, method='SLSQP'):
        # "context" seems to be some goal or target (see trajcost_true())
        # fed into self.reward_model.reward() network as context (see trajcost())
        res = minimize(lambda x: self.trajcost(reward_model, context, x),
                       self.xi0, method=method, constraints=self.constraints, 
                        options={'eps': self.eps, 'maxiter': self.max_iter}
        )

        xi = res.x.reshape(self.n_waypoints, self.state_dim)
        return xi

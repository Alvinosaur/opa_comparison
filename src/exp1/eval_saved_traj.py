"""
exp1 involves ... Thus, we define regret in this scenario as ...
This file should be called on each of the final saved trajectories of FERL, OPA, Unified, and 2017 paper to quantiatively compare their performance. 
"""

# TODO: something like this but without the if statements since we will just create separate folders for each experiment.
# trajectory reward function


def reward(self, xi, args, task=None):
    self.orientation = np.array([3.09817065, -0.053698958, -0.01449647])
    self.laptop = np.array([0.5, -0.35])
    R = 0
    self.args = args
    self.task = task
    if self.task == 'table':
        for idx in range(len(xi) - 1):
            R -= 1. * np.linalg.norm(xi[idx, 6:9] - xi[idx, :3]) * 2
            R -= 1.2 * abs(xi[idx, 2])
            if xi[idx, 2] < 0.08:
                R += 1 * xi[idx, 2]
    elif self.task == 'cup':
        for idx in range(len(xi) - 1):
            # z_tilt = xi[idx, 5:6]
            reward_pos = -np.linalg.norm(xi[idx, 6:12] - xi[idx, :6])
            reward_ang = -np.linalg.norm(self.orientation - xi[idx, 3:6])
            R += 1.0 * reward_pos + 0.75 * reward_ang
    elif self.task == 'laptop':
        for idx in range(len(xi) - 1):
            dist_laptop = np.linalg.norm(xi[idx, :2] - self.laptop)
            reward_target = -np.linalg.norm(xi[idx, :3] - xi[idx, 6:9])
            reward_laptop = -np.max([0.0, 1.0 - dist_laptop])
            R += 1 * reward_target + 1.0 * reward_laptop
    return R

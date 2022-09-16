import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import os

from globals import *


def save_tikz_dat_file(fpath, array):
    assert len(array.shape) > 1
    array = np.hstack([
        np.arange(array.shape[0])[:, np.newaxis],
        array
    ])
    np.savetxt(fpath, array)

all_data = []
methods = ["opa", "unified", "ferl", "online(oracle)", "online(missing)"]
folders = [
    "opa_saved_trials_inspection/eval_perturbs_1_time_60.0",
    "unified_saved_trials_inspection/eval_perturbs_10_time_100.0",
    "ferl_saved_trials_inspection_saved_eval/eval_perturbs_10_time_300.0/",

    "online_is_expert_False_saved_trials_inspection/eval_perturbs_10_time_30.0",
    "online_is_expert_True_saved_trials_inspection/eval_perturbs_10_time_30.0",
]
# 1s, 10s, 30s, 60s, 100s, 300s
# 
for method, folder in zip(methods, folders):
    metrics_data = np.load(os.path.join(folder, "metrics.npz"))
    all_pos_costs = metrics_data["all_pos_costs"]
    all_rot_costs = metrics_data["all_rot_costs"]

    # get original pos loss avg, std
    # 0th row is 0th exp_iter where perturb happened
    orig_pos_loss_avg = np.mean(all_pos_costs[0, :])
    orig_pos_loss_std = np.std(all_pos_costs[0, :])
    new_pos_loss_avg = np.mean(all_pos_costs[1:, :])
    new_pos_loss_std = np.std(all_pos_costs[1:, :])

    orig_rot_loss_avg = np.mean(all_rot_costs[0, :])
    orig_rot_loss_std = np.std(all_rot_costs[0, :])
    new_rot_loss_avg = np.mean(all_rot_costs[1:, :])
    new_rot_loss_std = np.std(all_rot_costs[1:, :])

    all_data.append([orig_pos_loss_avg, orig_pos_loss_std, new_pos_loss_avg, new_pos_loss_std, orig_rot_loss_avg, orig_rot_loss_std, new_rot_loss_avg, new_rot_loss_std])

# save to proper tikz format

save_tikz_dat_file("bar_plot_data.dat", np.array(all_data))
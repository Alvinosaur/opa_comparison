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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials_folder', action='store',
                        type=str, help="saved trial trajectory folder")
    args = parser.parse_args()

    return args

args = parse_arguments()

all_data = []
methods = ["opa", "unified", "ferl", "online"]

# expertfolders = [
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_1_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_2_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_3_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_4_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_5_time_120.0",
# ]

# missingfolders = [
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_1_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_2_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_3_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_4_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_5_time_120.0",
# ]

# unifiedfolders = [
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/eval_perturbs_1_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/eval_perturbs_2_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/eval_perturbs_3_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/eval_perturbs_4_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/eval_perturbs_5_time_120.0",
# ]

# opafolders = [
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/opa_saved_trials_inspection/eval_perturbs_1_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/opa_saved_trials_inspection/eval_perturbs_2_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/opa_saved_trials_inspection/eval_perturbs_3_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/opa_saved_trials_inspection/eval_perturbs_4_time_120.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/opa_saved_trials_inspection/eval_perturbs_5_time_120.0",
# ]

# ferlfolders = [
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection_final/eval_perturbs_1_time_30.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection_final/eval_perturbs_2_time_30.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection_final/eval_perturbs_3_time_30.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection_final/eval_perturbs_4_time_30.0",
#     "/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection_final/eval_perturbs_5_time_30.0",
# ]




expertfolders = [
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_10_time_1.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_10_time_10.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_10_time_30.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_10_time_60.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_True_saved_trials_inspection/eval_perturbs_10_time_120.0",
]

missingfolders = [
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_10_time_1.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_10_time_10.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_10_time_30.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_10_time_60.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/online_is_expert_False_saved_trials_inspection/eval_perturbs_10_time_120.0",
]

unifiedfolders = [
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/eval_perturbs_10_time_1.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/eval_perturbs_10_time_10.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/eval_perturbs_10_time_30.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/eval_perturbs_10_time_60.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/unified_saved_trials_inspection/eval_perturbs_10_time_120.0",
]

opafolders = [
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/opa_saved_trials_inspection/eval_perturbs_10_time_1.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/opa_saved_trials_inspection/eval_perturbs_10_time_10.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/opa_saved_trials_inspection/eval_perturbs_10_time_30.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/opa_saved_trials_inspection/eval_perturbs_10_time_60.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/opa_saved_trials_inspection/eval_perturbs_10_time_120.0",
]

ferlfolders = [
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection_final/eval_perturbs_10_time_1.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection_final/eval_perturbs_10_time_10.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection_final/eval_perturbs_10_time_30.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection_final/eval_perturbs_10_time_60.0",
    "/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection_final/eval_perturbs_10_time_300.0",
]


# # adaptation time
# for i in range(len(unifiedfolders)):
#     expert_metrics_data = np.load(os.path.join(expertfolders[i], "metrics.npz"))
#     missing_metrics_data = np.load(os.path.join(missingfolders[i], "metrics.npz"))
#     unified_metrics_data = np.load(os.path.join(unifiedfolders[i], "metrics.npz"))
#     opa_metrics_data = np.load(os.path.join(opafolders[i], "metrics.npz"))
#     ferl_metrics_data = np.load(os.path.join(ferlfolders[i], "metrics.npz"))

#     all_data.append([
#         np.mean(expert_metrics_data["all_pos_costs"][:, :]), 
#         np.std(expert_metrics_data["all_pos_costs"][:, :]),
#         np.mean(missing_metrics_data["all_pos_costs"][:, :]), 
#         np.std(missing_metrics_data["all_pos_costs"][:, :]),
#         np.mean(opa_metrics_data["all_pos_costs"][:, :]), 
#         np.std(opa_metrics_data["all_pos_costs"][:, :]),
#         np.mean(unified_metrics_data["all_pos_costs"][:, :]), 
#         np.std(unified_metrics_data["all_pos_costs"][:, :]),
#         np.mean(ferl_metrics_data["all_pos_costs"][:, :]), 
#         np.std(ferl_metrics_data["all_pos_costs"][:, :]),
#         ])

# generalization
expert_metrics_data = np.load(os.path.join(expertfolders[-1], "metrics.npz"))
missing_metrics_data = np.load(os.path.join(missingfolders[-1], "metrics.npz"))
unified_metrics_data = np.load(os.path.join(unifiedfolders[-1], "metrics.npz"))
opa_metrics_data = np.load(os.path.join(opafolders[-1], "metrics.npz"))
ferl_metrics_data = np.load(os.path.join(ferlfolders[-1], "metrics.npz"))

all_data.append([
        np.mean(expert_metrics_data["all_pos_costs"][0, :]), 
        np.std(expert_metrics_data["all_pos_costs"][0, :]),
        np.mean(expert_metrics_data["all_pos_costs"][1, :]), 
        np.std(expert_metrics_data["all_pos_costs"][1, :]),
        np.mean(expert_metrics_data["all_rot_costs"][0, :]), 
        np.std(expert_metrics_data["all_rot_costs"][0, :]),
        np.mean(expert_metrics_data["all_rot_costs"][1, :]), 
        np.std(expert_metrics_data["all_rot_costs"][1, :]),])
all_data.append([
        np.mean(missing_metrics_data["all_pos_costs"][0, :]), 
        np.std(missing_metrics_data["all_pos_costs"][0, :]),
        np.mean(missing_metrics_data["all_pos_costs"][1, :]), 
        np.std(missing_metrics_data["all_pos_costs"][1, :]),
        np.mean(missing_metrics_data["all_rot_costs"][0, :]), 
        np.std(missing_metrics_data["all_rot_costs"][0, :]),
        np.mean(missing_metrics_data["all_rot_costs"][1, :]), 
        np.std(missing_metrics_data["all_rot_costs"][1, :]),])
all_data.append([
        np.mean(opa_metrics_data["all_pos_costs"][0, :]), 
        np.std(opa_metrics_data["all_pos_costs"][0, :]),
        np.mean(opa_metrics_data["all_pos_costs"][1, :]), 
        np.std(opa_metrics_data["all_pos_costs"][1, :]),
        np.mean(opa_metrics_data["all_rot_costs"][0, :]), 
        np.std(opa_metrics_data["all_rot_costs"][0, :]),
        np.mean(opa_metrics_data["all_rot_costs"][1, :]), 
        np.std(opa_metrics_data["all_rot_costs"][1, :]),])
all_data.append([
        np.mean(unified_metrics_data["all_pos_costs"][0, :]), 
        np.std(unified_metrics_data["all_pos_costs"][0, :]),
        np.mean(unified_metrics_data["all_pos_costs"][1, :]), 
        np.std(unified_metrics_data["all_pos_costs"][1, :]),
        np.mean(unified_metrics_data["all_rot_costs"][0, :]), 
        np.std(unified_metrics_data["all_rot_costs"][0, :]),
        np.mean(unified_metrics_data["all_rot_costs"][1, :]), 
        np.std(unified_metrics_data["all_rot_costs"][1, :]),])
all_data.append([
        np.mean(ferl_metrics_data["all_pos_costs"][0, :]), 
        np.std(ferl_metrics_data["all_pos_costs"][0, :]),
        np.mean(ferl_metrics_data["all_pos_costs"][1, :]), 
        np.std(ferl_metrics_data["all_pos_costs"][1, :]),
        np.mean(ferl_metrics_data["all_rot_costs"][0, :]), 
        np.std(ferl_metrics_data["all_rot_costs"][0, :]),
        np.mean(ferl_metrics_data["all_rot_costs"][1, :]), 
        np.std(ferl_metrics_data["all_rot_costs"][1, :]),])

# save to proper tikz format

save_tikz_dat_file("bar_plot_data.dat", np.array(all_data))
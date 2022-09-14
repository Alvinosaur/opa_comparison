import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from ferl_network import DNN
import globals

net = DNN(nb_layers=3, nb_units=128, input_dim=12)
net.load_state_dict(torch.load("/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection/eval_perturbs_10_time_300.0/model_0.pth"))

traj = np.load("/home/ruic/Documents/opa/opa_comparison/src/exp1/ferl_saved_trials_inspection/perturb_collection/perturb_traj_iter_0_num_0.npy")

traj_euler = np.concatenate([traj[:,:3], R.from_quat(traj[:,3:]).as_euler("XYZ")], axis=-1)

human = np.concatenate([globals.inspection_poses[0], R.from_quat(globals.inspection_ori_quats[0]).as_euler("XYZ")])

traj_euler = np.hstack([traj_euler, np.repeat(human[np.newaxis, :], traj_euler.shape[0], axis=0)])

rewards = net(torch.from_numpy(traj_euler).to(torch.float32).to("cuda"))

print(rewards.detach().cpu().numpy())
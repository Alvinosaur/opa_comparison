import numpy as np
from trajopt import TrajOpt
from expert_demos import TrajOpt_Expert
from train_reward import TrainReward
from torch.utils.tensorboard import SummaryWriter
import datetime
import matplotlib.pyplot as plt
import os
import sys
import pickle
import torch
from scipy.spatial.transform import Rotation as R

import compare_exp1_globals as exp1

# Hyperparameters
state_len = 12  # fixed number of wayponts for trajopt

# Constants
reward_model1_path = "models/reward_model_1"
reward_model2_path = "models/reward_model_2"

# function for drawing the trajectory
def draw_interaction(ax, xi, alpha):
    ax.plot(xi[:, 0], xi[:, 1], 'bo-', alpha=alpha)
    ax.plot(range(10), xi[:, 2], 'ko-', alpha=alpha)
    ax.plot(range(10), xi[:,5], 'ro-', alpha = alpha)
    ax.plot(xi[0, 6], xi[0, 7], color='g', marker='s', markersize=10)
    # ax.plot(xi[0, 6], xi[0, 7], color='r', marker='s', markersize=10, alpha=alpha)
    ax.axis('equal')

# log and visualize the results
run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder = 'runs/{}_pHRI_{}'.format(run_name, "Orient")
writer = SummaryWriter(folder)
fig, ax = plt.subplots(1, 1)
if not os.path.exists("evals/" + run_name):
    os.makedirs("evals/" + run_name)

# initialize the reward models and trajopt
rm1 = TrainReward(save_path=reward_model1_path, noise=0.5, model_dim=(12, 128), epoch=500)
rm2 = TrainReward(save_path=reward_model2_path, noise=0.5, model_dim=(12, 128), epoch=500)

# load the expert demonstrations
demos = pickle.load(open("demos/expert_cup.pkl", "rb"),
        encoding='latin1')
demos_tensor = torch.FloatTensor(np.array(demos))

# states: [x, y, z, rx, ry, rz]
start_pose_aaxis = np.concatenate([
    exp1.start_pose_world[0:3],
    R.from_quat(exp1.start_pose_world[3:]).as_rotvec()
])
goal_pose_aaxis = np.concatenate([
    exp1.goal_pose_world[0:3],
    R.from_quat(exp1.goal_pose_world[3:]).as_rotvec()
])
trajopt = TrajOpt(home=start_pose_aaxis, 
                  goal=goal_pose_aaxis, 
                  state_len=state_len)
trajopt_expert = TrajOpt_Expert(home=start_pose_aaxis, 
                  goal=goal_pose_aaxis, 
                  state_len=state_len)

# train the models time on pre-collected data
Q = []
C = []
l1, l2 = np.inf, np.inf
while l1 > 0.3:
    l1 = rm1.train_rewards(demos, demos_tensor)
    print("final loss on model1: ", l1)
while l2 > 0.3:
    l2 = rm2.train_rewards(demos, demos_tensor)
    print("final loss on model2: ", l2)

# main loop: with different EE goals, examine behavior of model by comparing with expert trajopt
# Loop through different theta of planar 2D EE angle
THETA = np.linspace(0, np.pi, 5)
for iter_count in range(100):
    # evaluate the learned performance of model 1
    average_reward1 = 0
    for index in range(len(THETA)):
        # [x, y, z, rx, ry, rz]
        context = np.array([7*np.cos(THETA[index]), 7*np.sin(THETA[index]), 0., 0., 0., np.pi/2])

        xi1, _, _ = trajopt.optimize(rm1, context)
        average_reward1 += trajopt_expert.reward(xi1) / len(THETA)
        draw_interaction(ax, xi1, (index+1.)/len(THETA))
    plt.savefig("evals/" + run_name + "/iter_" + str(iter_count) + "_model1.png", dpi=100)
    ax.cla()

    # evaluate the learned performance of model 2
    average_reward2 = 0
    for index in range(len(THETA)):
        context = np.array([7*np.cos(THETA[index]), 7*np.sin(THETA[index]), 0., 0., 0., np.pi/2])
        xi2, _, _ = trajopt.optimize(rm2, context)
        average_reward2 += trajopt_expert.reward(xi2) / len(THETA)
        draw_interaction(ax, xi2, (index+1.)/len(THETA))
    plt.savefig("evals/" + run_name + "/iter_" + str(iter_count) + "_model2.png", dpi=100)
    ax.cla()

    # log performance
    print("Here is the average performance of each model on iteration: ", iter_count)
    print("Model 1:", average_reward1, "Model 2:", average_reward2)
    writer.add_scalar('loss/model1', average_reward1, iter_count)
    writer.add_scalar('loss/model2', average_reward2, iter_count)
    writer.add_scalar('loss/best', max([average_reward1, average_reward2]), iter_count)
    writer.add_scalar('data/preferences', len(Q), iter_count)
    writer.add_scalar('data/corrections', len(C), iter_count)

    # query the human / oracle
    # the queries seem super close -- I added a deformation
    theta = np.random.rand()*np.pi
    context = np.array([7*np.cos(theta), 7*np.sin(theta), 0., 0., 0., np.pi/2])
    xi1, _, _ = trajopt.optimize(rm1, context)
    xi2, _, _ = trajopt.optimize(rm2, context)
    xi1 = rm1.deform(xi1, np.random.normal(0, 0.5, 6))
    xi2 = rm2.deform(xi2, np.random.normal(0, 0.5, 6))
    R1 = trajopt_expert.reward(xi1)
    R2 = trajopt_expert.reward(xi2)
    alpha1, alpha2 = 0.3, 0.6
    if R2 < R1:
        alpha1, alpha2 = 0.6, 0.3
    draw_interaction(ax, xi1, alpha1)
    draw_interaction(ax, xi2, alpha2)

    # decide whether we want a query or correction
    if iter_count < 2:
        xi_corrected, _, _ = trajopt_expert.optimize(None, context)
        C.append(xi_corrected)
        draw_interaction(ax, xi_corrected, 1.0)
    else:
        if R1 > R2:
            query = [xi1, xi2]
        else:
            query = [xi2, xi1]
        Q.append(query)
    plt.savefig("evals/" + run_name + "/feedback_" + str(iter_count) + ".png", dpi=100)
    ax.cla()

    # if iter_count > 10:
    #     sys.exit() 

    # retrain the models picking up where we left off
    rm1.update_preferences(Q, C)
    rm2.update_preferences(Q, C)
    l1, l2 = np.inf, np.inf
    while l1 > 0.3:
        l1 = rm1.train_rewards(demos, demos_tensor, load_paths=reward_model1_path)
        print("final loss on model1: ", l1)
    while l2 > 0.3:
        l2 = rm2.train_rewards(demos, demos_tensor, load_paths=reward_model2_path)
        print("final loss on model2: ", l2)

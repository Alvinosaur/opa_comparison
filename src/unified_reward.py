import numpy as np
import torch
import copy
import os
from unified_model import StateFunction
from torch.optim import Adam
from torch.nn.utils.convert_parameters import parameters_to_vector
import matplotlib.pyplot as plt

from tqdm import tqdm


class TrainReward(object):

    def __init__(self, device, model_dim=(12, 128), action_dim=6, epoch=500, LR=0.001, L2_ratio=0.005, noise=2.0, n_models=3, traj_len=10,
                 batch_size=32):

        # hyperparameters
        self.device = device
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.EPOCH = epoch
        self.LR = LR
        self.L2_RATIO = L2_ratio
        self.noise = noise
        self.LR_STEP_SIZE = 400
        self.LR_GAMMA = 0.1
        self.n_models = n_models
        self.traj_len = traj_len
        self.batch_size = batch_size

        # ensemble to measure to uncertainty
        self.reward_models = [
            StateFunction(self.model_dim[0], self.model_dim[1]).to(device)
            for _ in range(n_models)
        ]

        # learning parameters
        self.loss_cal = torch.nn.CrossEntropyLoss()

        # precompute the deformations
        self.compute_deform_mats(length=2 * self.traj_len)

    def compute_deform_mats(self, length):
        A = np.zeros((length + 2, length))
        for idx in range(length):
            A[idx, idx] = 1
            A[idx + 1, idx] = -2
            A[idx + 2, idx] = 1
        self.R = np.linalg.inv(np.dot(A.T, A))
        self.U = np.zeros(length)

    # deform the trajectory to get a noisy version
    def deform(self, xi, tau, start_idx=None):
        xi_deformed = np.copy(xi)
        N = len(xi)
        self.compute_deform_mats(length=2 * N)
        if start_idx == None:
            start_idx = np.random.randint(1, N - 1)
        gamma = np.zeros((len(self.U), self.action_dim))
        for idx in range(self.action_dim):
            self.U[0] = tau[idx]
            # gamma[:,idx] = self.R @ self.U
            gamma[:, idx] = np.dot(self.R, self.U)

        xi_deformed[start_idx:, :self.action_dim] += gamma[:N - start_idx, :]
        return xi_deformed

    def train_rewards(self, demos, context, load_paths=None):
        if load_paths is not None:
            for i, path in enumerate(load_paths):
                self.reward_models[i].load_state_dict(torch.load(path))

        demos_tensor = [torch.from_numpy(demo).to(
            self.device).to(torch.float32) for demo in demos]
        context_tensor = torch.from_numpy(
            context).to(self.device).to(torch.float32)
        avg_loss = sum([
            self.train_reward(demos, demos_tensor,
                              context_tensor, self.reward_models[i])
            for i in range(self.n_models)]) / self.n_models

        return avg_loss

    # train the reward model
    def train_reward(self, demos, demos_tensor, context_tensor, reward_model):
        optim = Adam(reward_model.parameters(), lr=self.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=self.LR_STEP_SIZE, gamma=self.LR_GAMMA)
        num_demos = len(demos)
        # always assume orig demo (idx=0) is better than noisily perturbed demo
        base_labels = torch.zeros(num_demos, device=self.device)
        for epoch in tqdm(range(self.EPOCH)):
            logits_1 = torch.Tensor([])
            logits_2 = torch.Tensor([])

            for _ in range(self.batch_size):
                demo_idx = np.random.randint(0, num_demos)
                demo_orig = demos[demo_idx].copy()
                demo_orig_tensor = demos_tensor[demo_idx]

                # generate another artificial demo by deforming with random noise
                rand_noise = np.random.normal(0, self.noise, self.action_dim)
                demo_deformed = self.deform(demo_orig, rand_noise)
                demo_deformed_tensor = torch.from_numpy(
                    demo_deformed).to(self.device).to(torch.float32)

                R_orig = torch.sum(reward_model(
                    torch.cat([demo_orig_tensor, context_tensor], dim=-1)
                )).unsqueeze(0)
                R_deformed = torch.sum(reward_model(
                    torch.cat([demo_deformed_tensor, context_tensor], dim=-1)
                )).unsqueeze(0)
                logits_1 = torch.cat((logits_1, R_orig), dim=0)
                logits_2 = torch.cat((logits_2, R_deformed), dim=0)

            logits_12 = torch.cat(
                (logits_1.unsqueeze(1), logits_2.unsqueeze(1)), dim=1)
            loss_cmp = self.loss_cal(logits_12, base_labels)
            loss_l2 = self.L2_RATIO * \
                parameters_to_vector(reward_model.parameters()).norm() ** 2
            loss_reward = loss_cmp + loss_l2

            optim.zero_grad()
            loss_reward.backward()
            optim.step()
            scheduler.step()

            if epoch % 100 == 0:
                print("Training epoch: ", epoch, "Loss: ", loss_reward.item())

        return loss_reward.item()

    # use the ensemble reward models
    def reward(self, state):
        return sum([
            reward_model(state).sum().item() for reward_model in self.reward_models]) / self.n_models

    def save(self, folder, name):
        for i in range(self.n_models):
            torch.save(self.reward_models[i].state_dict(),
                       os.path.join(folder, name + f"_{i}" + ".pth"))

    # likelihood of choosing option 1 or option 2
    def preference_probability(self, query, reward_model):
        states1 = torch.FloatTensor(query[0])
        states2 = torch.FloatTensor(query[1])
        R1 = torch.sum(reward_model(states1)).item()
        R2 = torch.sum(reward_model(states2)).item()
        p_R1 = np.exp(R1) / (np.exp(R1) + np.exp(R2))
        return [p_R1, 1.0 - p_R1]

    # information gain
    def information_gain(self, query):
        p_rms = [
            self.preference_probability(query, reward_model)
            for reward_model in self.reward_models]
        normalizer1 = sum(p_rms, lambda x, y: x[0] + y[0])
        normalizer2 = sum(p_rms, lambda x, y: x[1] + y[1])
        # ig1 = p_rm1[0] * np.log2(3.0 * p_rm1[0] / normalizer1) + \
        #     p_rm2[0] * np.log2(3.0 * p_rm2[0] / normalizer1) + \
        #     p_rm3[0] * np.log2(3.0 * p_rm3[0] / normalizer1)
        info_gain1 = sum([
            p_rm[0] * np.log2(3.0 * p_rm[0] / normalizer1)
            for p_rm in p_rms
        ])

        # ig2 = p_rm1[1] * np.log2(3.0 * p_rm1[1] / normalizer2) + \
        #     p_rm2[1] * np.log2(3.0 * p_rm2[1] / normalizer2) + \
        #     p_rm3[1] * np.log2(3.0 * p_rm3[1] / normalizer2)
        info_gain2 = sum([
            p_rm[1] * np.log2(3.0 * p_rm[1] / normalizer2)
            for p_rm in p_rms
        ])
        # return 1.0 / 3 * (info_gain1 + info_gain2)
        return (info_gain1 + info_gain2) / 3

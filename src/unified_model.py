import torch
import torch.nn as nn
import torch.nn.functional as F


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class StateFunction(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(StateFunction, self).__init__()

        self.ReLU = nn.LeakyReLU()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):

        # to scale the laptop position into a more easily recognized range
        state = torch.clone(state)
        state[:,6] *= 10.0
        
        h1 = self.ReLU(self.linear1(state))
        h2 = self.ReLU(self.linear2(h1))
        return torch.tanh(self.linear3(h2))

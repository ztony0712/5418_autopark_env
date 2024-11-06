# autopark_env/models/ddpg_net.py
import torch
import torch.nn as nn

class DDPGNet(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(DDPGNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.action_output = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action = torch.tanh(self.action_output(x)) * self.max_action
        return action
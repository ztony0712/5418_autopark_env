# autopark_env/models/ddpg.py
import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(256, 256)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.action_output = nn.Linear(256, action_dim)
        nn.init.xavier_uniform_(self.action_output.weight)
        self.max_action = max_action
        
    def forward(self, state):
        # Process input of Dict type
        if isinstance(state, dict):
            state = state['achieved_goal']
        # Ensure input is a tensor and on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        # Move to the same device as the model
        state = state.to(self.fc1.weight.device)
            
        # Add dimension for BatchNorm1d
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = torch.relu(self.fc1(state))  # shape: (batch_size, 256)
        x = torch.relu(self.fc2(x))      # shape: (batch_size, 256)
        action = torch.tanh(self.action_output(x)) * self.max_action  # shape: (batch_size, action_dim)
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q_output = nn.Linear(256, 1)
                
    def forward(self, state, action):
        # Process input of Dict type
        if isinstance(state, dict):
            state = state['achieved_goal']
        # Ensure input is a tensor and on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action)
            
        # Move to the same device as the model
        device = self.fc1.weight.device
        state = state.to(device)
        action = action.to(device)
            
        # Add dimension for BatchNorm1d
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        sa = torch.cat([state, action], dim=1)  # shape: (batch_size, state_dim + action_dim)
        
        x = torch.relu(self.fc1(sa))  # shape: (batch_size, 256)
        x = torch.relu(self.fc2(x))   # shape: (batch_size, 256)
        q_value = self.q_output(x)    # shape: (batch_size, 1)
        
        return q_value
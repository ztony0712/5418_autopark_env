import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.action_output = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = self.max_action * torch.tanh(self.action_output(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 network
        self.fc1_q1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_q1 = nn.Linear(256, 256)
        self.q1_output = nn.Linear(256, 1)
        
        # Q2 network (TD3's unique double Q structure)
        self.fc1_q2 = nn.Linear(state_dim + action_dim, 256)
        self.fc2_q2 = nn.Linear(256, 256)
        self.q2_output = nn.Linear(256, 1)

    def forward(self, state, action):
        # Process Dict type input
        if isinstance(state, dict):
            state = state['achieved_goal']
        # Ensure input is a tensor and on the correct device
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action)
            
        # Move to the correct device
        device = self.fc1_q1.weight.device
        state = state.to(device)
        action = action.to(device)
        
        # Add dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        sa = torch.cat([state, action], 1)
        
        # Q1
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.q1_output(q1)
        
        # Q2
        q2 = F.relu(self.fc1_q2(sa))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.q2_output(q2)
        
        return q1, q2

    def Q1(self, state, action):
        # Process input as above
        if isinstance(state, dict):
            state = state['achieved_goal']
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action)
            
        device = self.fc1_q1.weight.device
        state = state.to(device)
        action = action.to(device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        sa = torch.cat([state, action], 1)
        
        q1 = F.relu(self.fc1_q1(sa))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.q1_output(q1)
        return q1

# autopark_env/models/ddpg.py
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_output = nn.Linear(128, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        # 处理Dict类型的输入
        if isinstance(state, dict):
            state = state['observation']
        # 确保输入是tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.action_output(x)) * self.max_action
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.q_output = nn.Linear(300, 1)
        
    def forward(self, state, action):
        # 处理Dict类型的输入
        if isinstance(state, dict):
            state = state['observation']
        # 确保输入是tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action)
            
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.q_output(x)
        return q_value
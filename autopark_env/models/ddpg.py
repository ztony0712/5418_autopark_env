# autopark_env/models/ddpg.py
import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        # 增加网络层数和神经元数量
        self.fc1 = nn.Linear(state_dim, 400)  # 第一层扩大到400
        self.ln1 = nn.LayerNorm(400)        # 添加LayerNorm
        
        self.fc2 = nn.Linear(400, 300)        # 第二层300个神经元
        self.ln2 = nn.LayerNorm(300)
        
        self.fc3 = nn.Linear(300, 200)        # 添加第三层
        self.ln3 = nn.LayerNorm(200)
        
        self.action_output = nn.Linear(200, action_dim)
        self.max_action = max_action
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """使用合适的初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.zero_()
        
    def forward(self, state):
        # 处理Dict类型的输入
        if isinstance(state, dict):
            state = state['observation']
        # 确保输入是tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
            
        # 添加维度，以适应BatchNorm1d
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        x = torch.relu(self.ln3(self.fc3(x)))
        action = torch.tanh(self.action_output(x)) * self.max_action
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # 第一个Q网络
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        
        self.fc2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        
        self.fc3 = nn.Linear(300, 200)
        self.ln3 = nn.LayerNorm(200)
        
        self.q_output = nn.Linear(200, 1)
        
        # 第二个Q网络（双Q网络结构，类似TD3）
        self.fc4 = nn.Linear(state_dim + action_dim, 400)
        self.ln4 = nn.LayerNorm(400)
        
        self.fc5 = nn.Linear(400, 300)
        self.ln5 = nn.LayerNorm(300)
        
        self.fc6 = nn.Linear(300, 200)
        self.ln6 = nn.LayerNorm(200)
        
        self.q_output2 = nn.Linear(200, 1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """使用合适的初始化方法"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.bias.data.zero_()
                
    def forward(self, state, action):
        # 处理Dict类型的输入
        if isinstance(state, dict):
            state = state['observation']
        # 确保输入是tensor
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action)
            
        # 添加维度，以适应BatchNorm1d
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        sa = torch.cat([state, action], dim=1)
        
        # 第一个Q值
        q1 = torch.relu(self.ln1(self.fc1(sa)))
        q1 = torch.relu(self.ln2(self.fc2(q1)))
        q1 = torch.relu(self.ln3(self.fc3(q1)))
        q1 = self.q_output(q1)
        
        # 第二个Q值
        q2 = torch.relu(self.ln4(self.fc4(sa)))
        q2 = torch.relu(self.ln5(self.fc5(q2)))
        q2 = torch.relu(self.ln6(self.fc6(q2)))
        q2 = self.q_output2(q2)
        
        # 返回两个Q值中的最小值（类似TD3的做法）
        return torch.min(q1, q2)
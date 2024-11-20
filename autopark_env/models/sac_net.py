import torch as th
from torch import nn
import numpy as np

class SACNet(nn.Module):
    """
    一个轻量级的策略网络，用于模仿SAC的行为
    Input shape: (batch_size, obs_dim)
    Output shape: (batch_size, action_dim)
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        
        # 特征提取器
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),  # 添加归一化层
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        
        # 策略头
        self.policy_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
            
    def forward(self, obs):
        features = self.feature_net(obs)
        return self.policy_net(features)

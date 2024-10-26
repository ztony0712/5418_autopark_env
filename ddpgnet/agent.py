import numpy as np
import torch
import torch.optim as optim
from .model import Actor, Critic
from .replay_buffer import ReplayBuffer

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer(max_size=100000)
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005   # 软更新目标网络的系数

        self.update_target_networks()

    def update_target_networks(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def select_action(self, state, noise=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        action += np.random.normal(0, noise, size=action.shape)  # 添加噪声以促进探索
        return np.clip(action, -1, 1)

    def train(self, batch_size):
        if self.replay_buffer.size() < batch_size:
            return

        # 从回放池中采样
        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # 更新 Critic 网络
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_values = self.target_critic(next_states, target_actions)
            targets = rewards + (1 - dones) * self.gamma * target_values

        critic_values = self.critic(states, actions)
        critic_loss = torch.nn.MSELoss()(critic_values, targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 Actor 网络
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.update_target_networks()

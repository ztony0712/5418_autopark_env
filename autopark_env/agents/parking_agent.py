# autopark_env/agents/parking_agent.py
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from autopark_env.models.ddpg import Actor, Critic
from autopark_env.models.replay_buffer import ReplayBuffer

class ParkingAgent:
    def __init__(self, env_name='my-new-env-v0', learning_rate=0.001, gamma=0.99, max_timesteps=1000):
        # 初始化环境和参数
        self.env = gym.make(env_name)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_timesteps = max_timesteps
        self.batch_size = 64
        self.exploration_noise = 0.1
        self.exploration_noise_decay = 0.995  # 添加探索噪声衰减
        self.tau = 0.005  # 目标网络软更新系数

        # 初始化模型和经验回放
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        self.actor = Actor(state_dim, action_dim, max_action)
        self.critic = Critic(state_dim, action_dim)

        # 初始化目标网络
        self.target_actor = Actor(state_dim, action_dim, max_action)
        self.target_critic = Critic(state_dim, action_dim)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.replay_buffer = ReplayBuffer(100000, state_dim, action_dim)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
         
    def train(self, num_episodes=100):
        # 训练主循环
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0

            for t in range(self.max_timesteps):
                # 选择动作并添加探索噪声
                action = self.actor(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
                noise = np.random.normal(0, self.exploration_noise, size=self.env.action_space.shape[0])
                action = action + noise
                # 动作裁剪，确保在合法范围内
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                # 与环境交互
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward

                # 存储经验
                self.replay_buffer.add(state, action, reward, next_state, float(done))
                state = next_state

                # 更新网络
                if self.replay_buffer.size > self.batch_size:
                    self.update_network()

                if done:
                    break

            # 衰减探索噪声
            self.exploration_noise *= self.exploration_noise_decay

            print(f"Episode {episode + 1}, Reward: {episode_reward}")
        
    def update_network(self):
        # 更新神经网络
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 转换为张量
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # [batch_size, 1]
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)      # [batch_size, 1]

        # 计算目标Q值，使用目标网络
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_Q = self.target_critic(next_states, next_actions)  # [batch_size, 1]

        # 确保 target_Q 的形状为 [batch_size, 1]
        target_Q = target_Q.view(-1, 1)
        
        # 打印形状
        # print('rewards shape:', rewards.shape)
        # print('dones shape:', dones.shape)
        # print('target_Q shape:', target_Q.shape)

        # 计算 target_Q
        target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # 计算当前Q值
        current_Q = self.critic(states, actions)          # [batch_size, 1]

        # 确保 current_Q 的形状为 [batch_size, 1]
        current_Q = current_Q.view(-1, 1)
        
        # 打印形状
        # print('current_Q shape:', current_Q.shape)
        # print('target_Q shape after calculation:', target_Q.shape)

        # 计算Critic损失
        critic_loss = F.mse_loss(current_Q, target_Q)

        # 更新Critic网络
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 计算Actor损失
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # 更新Actor网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)


    def soft_update(self, net, target_net):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, path='saved_models/'):
        torch.save(self.actor.state_dict(), f'{path}/actor.pth')
        torch.save(self.critic.state_dict(), f'{path}/critic.pth')
        print("Model saved.")

    def load_model(self, path='saved_models/'):
        self.actor.load_state_dict(torch.load(f'{path}/actor.pth', weights_only=True))
        self.critic.load_state_dict(torch.load(f'{path}/critic.pth', weights_only=True))
        print("Model loaded.")

    def test(self, num_episodes=10):
        self.load_model()
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.actor(torch.FloatTensor(state).unsqueeze(0)).detach().numpy()[0]
                # 动作裁剪
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                state, reward, done, _, _ = self.env.step(action)
                self.env.render()

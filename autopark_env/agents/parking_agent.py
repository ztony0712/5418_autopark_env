# autopark_env/agents/parking_agent.py
import time
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 400),
            torch.nn.LayerNorm(400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 300),
            torch.nn.LayerNorm(300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, action_dim),
            torch.nn.Tanh()
        )
        
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.q1_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, 400),
            torch.nn.LayerNorm(400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 300),
            torch.nn.LayerNorm(300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 1)
        )
        
        # Q2 architecture
        self.q2_net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, 400),
            torch.nn.LayerNorm(400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 300),
            torch.nn.LayerNorm(300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1_net(sa), self.q2_net(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1_net(sa)

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, state_dim))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.reward[ind],
            self.next_state[ind],
            self.done[ind]
        )

    def __len__(self):
        """返回当前buffer中的样本数量"""
        return self.size

class ParkingAgent:
    def __init__(self, env_name='my-new-env-v0', learning_rate=3e-4, gamma=0.99, max_timesteps=1000, batch_size=256):
        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 环境初始化
        self.env = gym.make(env_name)
        self.max_timesteps = max_timesteps
        self.batch_size = batch_size
        self.gamma = gamma

        # 获取状态和动作维度
        state_dim = self.env.observation_space['observation'].shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        # 修改探索参数
        self.exploration_noise = 0.5  # 增大初始探索噪声
        self.exploration_decay = 0.999  # 减缓探索噪声的衰减
        self.min_exploration_noise = 0.1  # 提高最小探索噪声，确保持续探索
        
        # 修改TD3特定参数
        self.tau = 0.005  # 恢复原始tau值
        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 1
        
        # 添加经验回放优先级
        self.min_buffer_size = 5000  # 开始训练前的最小经验数量
        self.reward_scale = 1.0  # 奖励缩放因子

        # 初始化网络
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 调整学习率
        learning_rate = 1e-4  # 原来是 3e-4，降低学习率
        
        # 初始化优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # 使用新的ReplayBuffer替换简单的列表
        self.replay_buffer = ReplayBuffer(
            state_dim=self.env.observation_space['observation'].shape[0],
            action_dim=self.env.action_space.shape[0],
            max_size=int(2e6)
        )

        # 添加新的参数
        self.success_threshold = 0.5  # 成功判定的距离阈值
        self.min_training_steps = 1000  # 开始训练前的最小步数

        # Tensorboard设置
        self.writer = SummaryWriter(f'runs/TD3_parking_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        
        # 训练统计
        self.total_steps = 0
        self.best_reward = float('-inf')

    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state_tensor).cpu().data.numpy().flatten()
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    def train(self, num_episodes=1000, save_interval=1000, test_env=None):
        """训练函数"""
        start_time = time.time()
        success_count = 0
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state_dict, _ = self.env.reset()
            state = state_dict['observation']
            episode_reward = 0
            episode_length = 0
            
            for t in range(self.max_timesteps):
                # 选择动作并添加探索噪声
                action = self.select_action(state)
                noise = np.random.normal(0, self.exploration_noise, size=action.shape)
                noisy_action = action + noise
                noisy_action = np.clip(noisy_action, self.env.action_space.low, self.env.action_space.high)
                
                # 执行动作
                next_state_dict, reward, terminated, truncated, info = self.env.step(noisy_action)
                next_state = next_state_dict['observation']
                done = terminated or truncated
                
                # 检查是否成功到达目标
                is_success = info.get('is_success', False)
                if is_success:
                    success_count += 1
                
                # 将 is_success 和 reward 记录到 TensorBoard
                self.writer.add_scalar('Training/Is_Success', int(is_success), self.total_steps)
                self.writer.add_scalar('Training/Step_Reward', reward, self.total_steps)
                
                # 存储经验
                self.replay_buffer.add(state, noisy_action, reward, next_state, done)
                
                # 更新网络
                if len(self.replay_buffer) > self.batch_size:
                    critic_loss, actor_loss = self.update_networks()
                    
                    # 记录损失
                    if critic_loss is not None:
                        self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.total_steps)
                    if actor_loss is not None:
                        self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.total_steps)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                self.total_steps += 1
                
                # 衰减探索噪声
                self.exploration_noise = max(
                    self.min_exploration_noise,
                    self.exploration_noise * self.exploration_decay
                )
                
                if done:
                    # 检查是否成功到达目标
                    is_success = info.get('is_success', False)
                    if is_success:
                        success_count += 1
                    break
            
            # 记录指标
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            success_rate = success_count / (episode + 1)
            self.writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
            self.writer.add_scalar('Training/Episode_Length', episode_length, episode)
            self.writer.add_scalar('Training/Success_Rate', success_rate, episode)
            
            # 打印训练信息
            if (episode + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | "
                      f"Success Rate: {success_rate:.2f} | "
                      f"Time: {elapsed_time:.2f}s")
    
    def test(self, env, num_episodes=5):
        """测试函数"""
        self.actor.eval()
        total_reward = 0
        
        for episode in range(num_episodes):
            state_dict, _ = env.reset()
            state = state_dict['observation']
            episode_reward = 0
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    action = self.actor(state_tensor).cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                next_state_dict, reward, terminated, truncated, info = env.step(action)
                state = next_state_dict['observation']
                episode_reward += reward
                done = terminated or truncated
                
            total_reward += episode_reward
            print(f"Test Episode {episode + 1}: Reward = {episode_reward}")
        
        avg_reward = total_reward / num_episodes
        self.actor.train()
        return avg_reward

    def update_networks(self):
        """更新网络"""
        # 采样batch
        batch = self.sample_batch()
        state, action, reward, next_state, done = batch

        # 更新Critic
        with torch.no_grad():
            # 目标策略平滑
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(self.env.action_space.low[0], self.env.action_space.high[0])
            
            # 使用双Q网络中的最小值
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            # 增加奖励缩放
            target_Q = reward * 10 + (1 - done) * self.gamma * target_Q

        # 当前Q值
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)  # 梯度裁剪
        self.critic_optimizer.step()

        actor_loss = None
        # 延迟策略更新
        if self.total_steps % self.policy_freq == 0:
            # 更新Actor
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)  # 梯度裁剪
            self.actor_optimizer.step()

            # 软更新目标网络
            self.soft_update(self.critic, self.critic_target)
            self.soft_update(self.actor, self.actor_target)
        
        return critic_loss, actor_loss

    def sample_batch(self):
        """从经验回放缓冲区采样"""
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # 转换为tensor并移动到正确的设备上
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)
        
        return state, action, reward, next_state, done

    def soft_update(self, local_model, target_model):
        """软更新目标网络"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, path='saved_models'):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f'{path}/td3_actor.pth')
        torch.save(self.critic.state_dict(), f'{path}/td3_critic.pth')
        print(f"Model saved to {path}")

    def load_model(self, path='saved_models'):
        """加载模型"""
        self.actor.load_state_dict(torch.load(f'{path}/td3_actor.pth', map_location=self.device))
        self.critic.load_state_dict(torch.load(f'{path}/td3_critic.pth', map_location=self.device))
        print("Model loaded")

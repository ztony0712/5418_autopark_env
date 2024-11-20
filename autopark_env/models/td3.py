import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import time

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh()
        )
        
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Q1 architecture
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Q2 architecture
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1_net(sa), self.q2_net(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1_net(sa)

class TD3Agent:
    def __init__(self, env_name, learning_rate, gamma, max_timesteps, batch_size, device):
        self.device = device
        self.env = gym.make(env_name)
        
        state_dim = self.env.observation_space['observation'].shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])
        
        # TD3 hyperparameters
        self.gamma = gamma
        self.max_timesteps = max_timesteps
        self.batch_size = batch_size
        self.tau = 0.001
        self.policy_noise = 0.1 * max_action
        self.noise_clip = 0.25 * max_action
        self.policy_freq = 3
        
        # Exploration parameters
        self.exploration_noise = 0.3
        self.exploration_decay = 0.9995
        self.min_exploration_noise = 0.05
        
        # Gradient clipping parameters
        self.max_grad_norm = 1.0
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = []
        self.max_buffer_size = int(1e6)
        self.min_buffer_size = 5000
        
        # Initialize tensorboard
        self.writer = SummaryWriter(f'runs/TD3_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        
        self.total_steps = 0
        
    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        return self.actor(state).cpu().data.numpy()

    def train(self, num_episodes):
        start_time = time.time()
        success_count = 0
        episode_rewards = []
        episode_lengths = []
        
        # Pre-fill the replay buffer
        print("Pre-filling the replay buffer...")
        while len(self.replay_buffer) < self.min_buffer_size:
            state_dict, _ = self.env.reset()
            state = state_dict['observation']
            for t in range(self.max_timesteps):
                action = np.random.uniform(
                    low=self.env.action_space.low,
                    high=self.env.action_space.high
                )
                next_state_dict, reward, terminated, truncated, info = self.env.step(action)
                next_state = next_state_dict['observation']
                done = terminated or truncated
                
                self.replay_buffer.append((state, action, reward, next_state, done))
                if len(self.replay_buffer) > self.max_buffer_size:
                    self.replay_buffer.pop(0)
                    
                if done:
                    break
                state = next_state
        print(f"Replay buffer pre-filled, size: {len(self.replay_buffer)}")
        
        for episode in range(num_episodes):
            state_dict, _ = self.env.reset()
            state = state_dict['observation']
            episode_step = 0
            
            for t in range(self.max_timesteps):
                # Select action with noise for exploration
                action = self.select_action(state)
                noise = np.random.normal(0, self.exploration_noise, size=action.shape)
                noisy_action = action + noise
                noisy_action = np.clip(noisy_action, self.env.action_space.low, self.env.action_space.high)
                
                # Execute action
                next_state_dict, reward, terminated, truncated, info = self.env.step(noisy_action)
                next_state = next_state_dict['observation']
                done = terminated or truncated
                
                # Store transition
                self.replay_buffer.append((state, noisy_action, reward, next_state, done))
                if len(self.replay_buffer) > self.max_buffer_size:
                    self.replay_buffer.pop(0)
                
                # Update networks and record losses
                if len(self.replay_buffer) > self.batch_size:
                    critic_loss, actor_loss = self.update_networks()
                    
                    if critic_loss is not None:
                        self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.total_steps)
                    if actor_loss is not None:
                        self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.total_steps)
                
                state = next_state
                episode_step += 1
                self.total_steps += 1
                
                # Decay exploration noise
                self.exploration_noise = max(
                    self.min_exploration_noise,
                    self.exploration_noise * self.exploration_decay
                )
                
                if done:
                    # 检查是否达到目标
                    if info.get('is_success', False):
                        success_count += 1
                    break
            
            # 记录episode相关指标
            episode_rewards.append(reward)
            episode_lengths.append(episode_step)
            
            # 记录详细的训练指标
            self.writer.add_scalar('Training/Episode_Reward', reward, episode)
            self.writer.add_scalar('Training/Episode_Length', episode_step, episode)
            self.writer.add_scalar('Training/Success_Rate', success_count / (episode + 1), episode)
            self.writer.add_scalar('Training/Average_Reward', np.mean(episode_rewards[-100:]), episode)
            
            # 记录action相关指标
            self.writer.add_scalar('Actions/Mean', np.mean(action), episode)
            self.writer.add_scalar('Actions/Std', np.std(action), episode)
            
            # 记录buffer大小
            self.writer.add_scalar('Buffer/Size', len(self.replay_buffer), episode)
            
            # 每10个episodes打印信息
            if (episode + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Avg Length: {avg_length:.2f} | "
                      f"Success Rate: {success_count/(episode+1):.2f} | "
                      f"Buffer Size: {len(self.replay_buffer)} | "
                      f"Time: {elapsed_time:.2f}s")
    
    def update_networks(self):
        batch = self.sample_batch()
        state, action, reward, next_state, done = batch
        
        # Update critic
        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            
            next_action = self.actor_target(next_state) + noise
            next_action = next_action.clamp(self.env.action_space.low[0], self.env.action_space.high[0])
            
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q
        
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        actor_loss = None
        # Delayed policy updates
        if self.total_steps % self.policy_freq == 0:
            # Update actor
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # Update target networks
            self.soft_update(self.critic, self.critic_target)
            self.soft_update(self.actor, self.actor_target)
        
        return critic_loss, actor_loss
    
    def sample_batch(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[i] for i in indices]
        
        state = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        action = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        reward = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        done = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(1).to(self.device)
        
        return state, action, reward, next_state, done
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def save_model(self, path='saved_models'):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), f'{path}/td3_actor.pth')
        torch.save(self.critic.state_dict(), f'{path}/td3_critic.pth')
        print(f"Model saved to {path}")
    
    def load_model(self, path='saved_models'):
        self.actor.load_state_dict(torch.load(f'{path}/td3_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{path}/td3_critic.pth'))
        print("Model loaded")

# autopark_env/agents/parking_agent.py
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
from autopark_env.models.ddpg import Actor, Critic
from autopark_env.models.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
import os

class ParkingAgent:
    def __init__(self, env_name='my-new-env-v0', learning_rate=0.001, gamma=0.99, max_timesteps=1000):
        # Check if a GPU is available
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Initialize environment and parameters
        self.env = gym.make(env_name)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.max_timesteps = max_timesteps
        self.batch_size = 256
        self.exploration_noise = 1.0
        self.exploration_noise_decay = 0.999
        self.tau = 0.01

        # Get state and action dimensions
        state_dim = self.env.observation_space['observation'].shape[0]
        action_dim = self.env.action_space.shape[0]
        max_action = float(self.env.action_space.high[0])

        # Initialize networks and move to GPU
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        # Initialize experience replay buffer
        self.replay_buffer = ReplayBuffer(state_dim, action_dim)

        # Initialize Tensorboard
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.writer = SummaryWriter(f'runs/parking_agent_{current_time}')
        
        # Variables for tracking training progress
        self.total_steps = 0
        self.best_reward = float('-inf')

    def train(self, num_episodes=100):
        start_time = time.time()
        episode_rewards = []
        success_count = 0
        
        for episode in range(num_episodes):
            state_dict, _ = self.env.reset()
            state = state_dict['observation']
            episode_reward = 0
            episode_steps = 0
            episode_q_values = []
            episode_actor_losses = []
            episode_critic_losses = []
            
            # Get initial distance
            initial_distance = np.linalg.norm(
                state[:2] - self.env.unwrapped.parking_lot.goal_position
            )

            for t in range(self.max_timesteps):
                # Convert state to tensor and move to GPU
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Select action
                with torch.no_grad():
                    action = self.actor(state_tensor).cpu().numpy()[0]
                noise = np.random.normal(0, self.exploration_noise, size=self.env.action_space.shape[0])
                action = action + noise
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

                # Execute action
                next_state_dict, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = next_state_dict['observation']
                episode_reward += reward
                episode_steps += 1
                self.total_steps += 1

                # Store experience
                self.replay_buffer.add(state, action, reward, next_state, float(done))

                # Update networks
                if self.replay_buffer.size > self.batch_size:
                    q_value, actor_loss, critic_loss = self.update_network()
                    episode_q_values.append(q_value)
                    episode_actor_losses.append(actor_loss)
                    episode_critic_losses.append(critic_loss)

                state = next_state
                if done:
                    # Check if goal is reached
                    # if self.env.check_goal_reached(
                    #     next_state_dict['achieved_goal'], 
                    #     next_state_dict['desired_goal']
                    # ):
                    #     success_count += 1
                    # break
                
                    if self.env.unwrapped.check_goal_reached(
                        next_state_dict['achieved_goal'], 
                        next_state_dict['desired_goal']
                    ):
                        success_count += 1
                    break


            # Calculate current distance
            final_distance = np.linalg.norm(
                state[:2] - self.env.unwrapped.parking_lot.goal_position
            )

            # Log training metrics
            self.writer.add_scalar('Training/Episode Reward', episode_reward, episode)
            self.writer.add_scalar('Training/Episode Length', episode_steps, episode)
            self.writer.add_scalar('Training/Exploration Noise', self.exploration_noise, episode)
            
            # New metrics
            if episode_critic_losses:
                avg_critic_loss = sum(episode_critic_losses) / len(episode_critic_losses)
                self.writer.add_scalar('Training/Critic Loss', avg_critic_loss, episode)
            
            self.writer.add_scalar('Training/Distance to Goal', final_distance, episode)
            
            # Calculate and log success rate
            success_rate = success_count / (episode + 1)
            self.writer.add_scalar('Training/Success Rate', success_rate, episode)
            
            if episode_q_values:
                avg_q_value = sum(episode_q_values) / len(episode_q_values)
                avg_actor_loss = sum(episode_actor_losses) / len(episode_actor_losses)
                self.writer.add_scalar('Training/Average Q Value', avg_q_value, episode)
                self.writer.add_scalar('Training/Actor Loss', avg_actor_loss, episode)

            # Update best reward and save model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model()

            # Decay exploration noise
            self.exploration_noise = max(self.exploration_noise * self.exploration_noise_decay, 0.1)
            episode_rewards.append(episode_reward)

            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / 10
                elapsed_time = time.time() - start_time
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Best Reward: {self.best_reward:.2f} | "
                      f"Steps: {self.total_steps} | "
                      f"Time: {elapsed_time:.2f}s")

        self.writer.close()

    def update_network(self):
        # Sample from experience replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensor and move to GPU
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        # Critic loss
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Modify: return three values
        return current_Q.mean().item(), actor_loss.item(), critic_loss.item()

    def save_model(self, path='saved_models'):
        # Create directory (including all necessary parent directories)
        os.makedirs(path, exist_ok=True)
        
        # Save model
        torch.save(self.actor.state_dict(), f'{path}/actor.pth')
        torch.save(self.critic.state_dict(), f'{path}/critic.pth')
        print(f"Model saved to {path}")

    def load_model(self, path='saved_models'):
        self.actor.load_state_dict(torch.load(f'{path}/actor.pth', map_location=torch.device('cpu')))
        self.critic.load_state_dict(torch.load(f'{path}/critic.pth', map_location=torch.device('cpu')))
        print("Model loaded.")

    def test(self, num_episodes=10):
        # Set to evaluation mode
        self.actor.eval()  
        
        for episode in range(num_episodes):
            # Reset environment
            state, _ = self.env.reset()
            state = state['observation']
            episode_reward = 0
            done = False
            
            while not done:
                # Convert state to tensor and move to GPU
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action = self.actor(state_tensor).cpu().numpy()[0]
                
                # Action clipping
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state['observation']
                episode_reward += reward
                self.env.render()
                
            print(f"Test Episode {episode + 1}: Reward = {episode_reward}")
        
        # Restore training mode
        self.actor.train()  

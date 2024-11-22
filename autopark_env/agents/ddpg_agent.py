# autopark_env/agents/ddpg_agent.py
import torch
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import copy

from autopark_env.models.ddpg import Actor, Critic
from autopark_env.agents.base_agent import BaseAgent

class DDPGAgent(BaseAgent):
    def __init__(
        self,
        actor,
        critic,
        config,
        device=None
    ):
        super().__init__(config)
        
        # Get DDPG-specific parameters from config
        ddpg_config = config['ddpg']
        self.actor_lr = ddpg_config['actor_lr']
        self.critic_lr = ddpg_config['critic_lr']
        self.gamma = ddpg_config['gamma']
        self.tau = ddpg_config['tau']
        self.batch_size = ddpg_config['batch_size']
        self.buffer_size = ddpg_config['buffer_size']
        self.her_k = ddpg_config['her_k']
        self.future_p = ddpg_config['future_p']
        
        # Initialize networks
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        
        # Create target networks
        state_dim = actor.fc1.in_features
        action_dim = actor.action_output.out_features
        max_action = actor.max_action
        
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        
        # Copy parameters to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # Experience replay related
        self.episode_buffer = []
        
        # Exploration noise
        self.noise_scale = 0.1
        
        # Tensorboard
        self.writer = SummaryWriter(f'runs/DDPG_{datetime.now().strftime("%Y%m%d-%H%M%S")}')
        self.total_steps = 0
        
        # Add new metrics statistics
        self.episode_successes = []  # Used to calculate success rate
        self.success_window = 100    # Calculate success rate of the last 100 episodes
        
        # Add OU noise
        self.noise = OUNoise(action_dim, theta=0.15, sigma=0.2)
        
        # Add gradient clipping
        self.max_grad_norm = 1.0  # Consistent with TD3
        
        # Add exploration-related parameters
        self.initial_noise = 0.1
        self.final_noise = 0.05
        self.exploration_fraction = 0.8
        
    def select_action(self, state, add_noise=True):
        """Select action"""
        with torch.no_grad():
            # Process state dictionary, concatenate achieved_goal and desired_goal
            processed_state = np.concatenate((state['achieved_goal'], state['desired_goal']))  # shape: (12,)
            processed_state = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)  # shape: (1, 12)
            
            # Get action
            action = self.actor(processed_state).cpu().numpy().squeeze()  # shape: (action_dim,)
            
            if add_noise:
                noise = self.noise.sample() * self.exploration_noise  # Use OU noise
                action = np.clip(action + noise, -1, 1)
            return action
            
    def train(self, batch=None):
        """Train one step"""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # If no batch is provided, sample from experience replay
        if batch is None:
            batch = self.replay_buffer.sample(self.batch_size)
            
        # Unpack batch data, optimize data conversion method
        state_batch = torch.FloatTensor(np.array([
            np.concatenate((x[0]['achieved_goal'], x[0]['desired_goal'])) for x in batch
        ])).to(self.device)  # shape: (batch_size, 12)
        
        action_batch = torch.FloatTensor(np.array([x[1] for x in batch])).to(self.device)
        reward_batch = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array([
            np.concatenate((x[3]['achieved_goal'], x[3]['desired_goal'])) for x in batch
        ])).to(self.device)  # shape: (batch_size, 12)
        done_batch = torch.FloatTensor(np.array([x[4] for x in batch])).to(self.device)
        
        # Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_state_batch)
            target_q = self.critic_target(next_state_batch, next_actions)
            target_value = reward_batch + (1 - done_batch) * self.gamma * target_q
            
        current_q = self.critic(state_batch, action_batch)
        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # Update Actor
        actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        # Log losses
        self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.total_steps)
        self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.total_steps)
        
        # Log gradient norms
        self.writer.add_scalar('Gradients/Actor', actor_grad_norm, self.total_steps)
        self.writer.add_scalar('Gradients/Critic', critic_grad_norm, self.total_steps)
        
        # Log Q-value statistics
        self.writer.add_scalar('Q_values/Mean', current_q.mean().item(), self.total_steps)
        self.writer.add_scalar('Q_values/Max', current_q.max().item(), self.total_steps)
        self.writer.add_scalar('Q_values/Min', current_q.min().item(), self.total_steps)
        
        self.total_steps += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'q_value_mean': current_q.mean().item(),
            'actor_grad_norm': actor_grad_norm,
            'critic_grad_norm': critic_grad_norm
        }
        
    def add_to_buffer(self, state, action, reward, next_state, done, info=None):
        """Add experience to temporary buffer"""
        self.episode_buffer.append((state, action, reward, next_state, done, info))
            
    def _soft_update(self, local_model, target_model):
        """Soft update target network"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
    def save(self, path):
        """Save model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def update_success_rate(self, success):
        """Update and record success rate"""
        self.episode_successes.append(float(success))
        if len(self.episode_successes) > self.success_window:
            self.episode_successes.pop(0)
        success_rate = np.mean(self.episode_successes)
        return success_rate

    def process_episode(self):
        """Process the entire episode's data, applying HER strategy"""
        # Reset OU noise
        self.noise.reset()
        
        # Call the base class's process_episode to handle original experiences
        super().process_episode()
        
        # If needed, DDPG-specific episode processing logic can be added here

class OUNoise:
    """Ornstein-Uhlenbeck process"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset noise state"""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Generate noise sample"""
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

import torch
import torch.nn.functional as F
import numpy as np

from autopark_env.models.td3 import Actor, Critic
from autopark_env.agents.base_agent import BaseAgent

class TD3Agent(BaseAgent):
    def __init__(
        self,
        actor,
        critic,
        config,
        device=None
    ):
        super().__init__(config)
        
        # Get TD3-specific parameters from config
        td3_config = config['td3']
        self.actor_lr = td3_config['actor_lr']
        self.critic_lr = td3_config['critic_lr']
        self.gamma = td3_config['gamma']
        self.tau = td3_config['tau']
        self.policy_noise = td3_config['policy_noise']
        self.noise_clip = td3_config['noise_clip']
        self.policy_delay = td3_config['policy_delay']
        self.batch_size = td3_config['batch_size']
        self.buffer_size = td3_config['buffer_size']
        self.her_k = td3_config['her_k']
        self.future_p = td3_config['future_p']
        
        # Initialize networks
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        
        # Create target networks
        state_dim = actor.fc1.in_features  # achieved_goal (6) + desired_goal (6)
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
        self.total_it = 0
        
        # Exploration noise
        self.noise_scale = 0.1
        self.exploration_noise = self.noise_scale
        
        # Tensorboard
        self.total_steps = 0
        
    def select_action(self, state, add_noise=True):
        """Select action"""
        with torch.no_grad():
            # Process state dictionary, concatenate achieved_goal and desired_goal
            processed_state = np.concatenate((state['achieved_goal'], state['desired_goal']))  # shape: (12,)
            processed_state = torch.FloatTensor(processed_state).unsqueeze(0).to(self.device)  # shape: (1, 12)
            
            # Get action
            action = self.actor(processed_state).cpu().numpy().squeeze()  # shape: (action_dim,)
            
            if add_noise:
                noise = np.random.normal(0, self.exploration_noise, size=action.shape)
                action = np.clip(action + noise, -1, 1)
            return action
                
    def train(self, batch=None):
        """Train one step"""
        self.total_it += 1
        
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # If no batch is provided, sample from experience replay
        if batch is None:
            batch = self.replay_buffer.sample(self.batch_size)
            
        # Unpack batch data, optimize data conversion method
        state_batch = torch.FloatTensor(np.array([
            np.concatenate((x[0]['achieved_goal'], x[0]['desired_goal'])) for x in batch
        ])).to(self.device)  # shape: (batch_size, 12)
        
        action_batch = torch.FloatTensor(np.array([x[1] for x in batch])).to(self.device)  # shape: (batch_size, action_dim)
        reward_batch = torch.FloatTensor(np.array([x[2] for x in batch])).to(self.device)  # shape: (batch_size,)
        next_state_batch = torch.FloatTensor(np.array([
            np.concatenate((x[3]['achieved_goal'], x[3]['desired_goal'])) for x in batch
        ])).to(self.device)  # shape: (batch_size, 12)
        done_batch = torch.FloatTensor(np.array([x[4] for x in batch])).to(self.device)  # shape: (batch_size,)
        
        with torch.no_grad():
            # Target policy smoothing
            noise = torch.randn_like(action_batch) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_action = self.actor_target(next_state_batch) + noise
            next_action = torch.clamp(next_action, -1, 1)
            
            # Target Q value
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (1 - done_batch) * self.gamma * target_Q
        
        # Current Q value
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)
        
        # Critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        actor_loss = None
        actor_grad_norm = 0
        
        # Delayed policy update
        if self.total_it % self.policy_delay == 0:
            # Actor loss
            actor_actions = self.actor(state_batch)
            actor_loss = -self.critic.Q1(state_batch, actor_actions).mean()
            
            # Update Actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            # Soft update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
        
        # Log training information
        self.writer.add_scalar('Loss/Critic', critic_loss.item(), self.total_steps)
        if actor_loss is not None:
            self.writer.add_scalar('Loss/Actor', actor_loss.item(), self.total_steps)
        
        # Log Q value statistics
        q_values = torch.min(current_Q1, current_Q2)
        self.writer.add_scalar('Q_values/Mean', q_values.mean().item(), self.total_steps)
        self.writer.add_scalar('Q_values/Max', q_values.max().item(), self.total_steps)
        self.writer.add_scalar('Q_values/Min', q_values.min().item(), self.total_steps)
        self.writer.add_scalar('Q_values/Diff', (current_Q1 - current_Q2).abs().mean().item(), self.total_steps)
        
        # Log gradient norms
        self.writer.add_scalar('Gradients/Critic', critic_grad_norm, self.total_steps)
        if actor_grad_norm > 0:
            self.writer.add_scalar('Gradients/Actor', actor_grad_norm, self.total_steps)
            
        self.total_steps += 1
        
        return {
            'actor_loss': actor_loss.item() if actor_loss is not None else None,
            'critic_loss': critic_loss.item(),
            'q_value_mean': q_values.mean().item(),
            'q_diff_mean': (current_Q1 - current_Q2).abs().mean().item(),
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

    def process_episode(self):
        """Process the entire episode's data, applying HER strategy"""
        # Call the base class's process_episode to handle original experiences
        super().process_episode()
        
        # If needed, DDPG-specific episode processing logic can be added here

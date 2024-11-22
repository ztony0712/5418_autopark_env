import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
from collections import deque

class ReplayBuffer:
    """Experience replay buffer"""
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.position = 0
        
    def push(self, transition):
        """Add an experience"""
        if len(self.buffer) < self.max_size:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.max_size
        
    def sample(self, batch_size):
        """Randomly sample a batch"""
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        return [self.buffer[idx] for idx in indices]
        
    def __len__(self):
        return len(self.buffer)

class BaseAgent:
    """Base class for all agents, containing common methods"""
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Use the current time to create a writer
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f'runs/{self.__class__.__name__}_{current_time}')
        
        # Get parameters from the configuration file
        self.success_window = config['training']['success_window']
        self.episode_successes = deque(maxlen=self.success_window)
        
        # Get exploration parameters from the configuration file
        self.initial_noise = config['training']['initial_noise']
        self.final_noise = config['training']['final_noise']
        self.exploration_fraction = config['training']['exploration_fraction']
        self.exploration_noise = self.initial_noise
        
        # Initialize these attributes
        self.replay_buffer = ReplayBuffer(config['ddpg']['buffer_size'] 
            if 'ddpg' in config else config['td3']['buffer_size'])
        self.episode_buffer = []
        self.her_k = 4  # Default value, will be overridden by subclasses
        
    def select_action(self, state):
        """Select action"""
        raise NotImplementedError
        
    def train(self, batch=None):
        """Train one step"""
        raise NotImplementedError
        
    def add_to_buffer(self, state, action, reward, next_state, done, info=None):
        """Add experience to buffer"""
        raise NotImplementedError
        
    def process_episode(self):
        """Process the entire episode's data, applying HER strategy"""
        # Add original experiences
        for transition in self.episode_buffer:
            state, action, reward, next_state, done, info = transition
            self.replay_buffer.push((state, action, reward, next_state, done))
            
        # Apply HER strategy
        if self.her_k > 0 and len(self.episode_buffer) > 0:
            # Get all achieved goals in the episode as potential future goals
            achieved_goals = [t[3]['achieved_goal'] for t in self.episode_buffer]  # shape: (episode_len, goal_dim)
            
            # For each time step
            for t, transition in enumerate(self.episode_buffer):
                state, action, _, next_state, _, info = transition
                
                # Sample k future goals for each transition
                future_idx = np.random.randint(t, len(self.episode_buffer), size=self.her_k)
                
                # Create new virtual experiences using these future goals
                for idx in future_idx:
                    # Use the achieved goal at a future time as the new goal
                    new_goal = achieved_goals[idx].copy()
                    
                    # Create new state and next_state, replacing the goal
                    new_state = state.copy()
                    new_next_state = next_state.copy()
                    new_state['desired_goal'] = new_goal
                    new_next_state['desired_goal'] = new_goal
                    
                    # Access the underlying environment
                    base_env = info['env']
                    while hasattr(base_env, 'env'):
                        base_env = base_env.env
                    
                    # Calculate the new reward using the environment's compute_reward function
                    new_reward = base_env.compute_reward(
                        achieved_goal=next_state['achieved_goal'],
                        desired_goal=new_goal,
                        info=None
                    )
                    
                    # Add to the replay buffer
                    self.replay_buffer.push((new_state, action, new_reward, new_next_state, False))
            
            # Record the success of the original goal
            final_state = self.episode_buffer[-1][3]
            original_goal = self.episode_buffer[0][0]['desired_goal']
            
            # Unwrap the environment to get to the base environment
            base_env = info['env']
            while hasattr(base_env, 'env'):
                base_env = base_env.env
            
            original_success = base_env.check_goal_reached(
                final_state['achieved_goal'],
                original_goal
            )
            self.update_success_rate(float(original_success))
        
        # Clear the episode buffer
        self.episode_buffer = []
        
    def save(self, path):
        """Save model"""
        raise NotImplementedError
        
    def load(self, path):
        """Load model"""
        raise NotImplementedError
        
    def update_success_rate(self, success):
        """Update and record success rate"""
        self.episode_successes.append(float(success))
        if len(self.episode_successes) > self.success_window:
            self.episode_successes.pop(0)
        return np.mean(self.episode_successes)
        
    def update_exploration_noise(self, current_step, total_steps):
        """Update exploration noise"""
        exploration_steps = int(total_steps * self.exploration_fraction)
        if current_step <= exploration_steps:
            self.exploration_noise = self.initial_noise - \
                (self.initial_noise - self.final_noise) * \
                (current_step / exploration_steps)
        else:
            self.exploration_noise = self.final_noise

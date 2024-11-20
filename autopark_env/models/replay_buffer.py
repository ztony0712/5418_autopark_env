# autopark_env/models/replay_buffer.py
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # Initialize the buffer
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

        # Add new statistics
        self.positive_experiences = 0
        self.negative_experiences = 0
        self.reward_threshold = 0.0
        self.collision_count = 0

    def add(self, state, action, reward, next_state, done):
        # Update statistics
        if reward > self.reward_threshold:
            self.positive_experiences += 1
        elif reward < -self.reward_threshold:
            self.negative_experiences += 1
            if reward < -1.0:
                self.collision_count += 1
        
        # If collision ratio is too high, consider whether to store
        collision_ratio = self.collision_count / (self.size + 1) if self.size > 0 else 0
        if reward < -1.0 and collision_ratio > 0.3:
            return
            
        # Store
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # Ensure enough samples
        if self.size < batch_size:
            return None
            
        # Calculate positive and negative sample ratios
        pos_ratio = self.positive_experiences / self.size
        neg_ratio = self.negative_experiences / self.size
        
        # If sample ratio is severely unbalanced, perform balanced sampling
        if pos_ratio < 0.2 or neg_ratio < 0.2:
            # Sample positive and negative samples separately
            pos_indices = np.where(self.reward[:self.size] > self.reward_threshold)[0]
            neg_indices = np.where(self.reward[:self.size] < -self.reward_threshold)[0]
            neutral_indices = np.where(np.abs(self.reward[:self.size]) <= self.reward_threshold)[0]
            
            # Determine the number of samples for each type
            pos_samples = max(batch_size // 3, int(batch_size * 0.2))
            neg_samples = max(batch_size // 3, int(batch_size * 0.2))
            neutral_samples = batch_size - pos_samples - neg_samples
            
            # Sample indices
            indices = np.concatenate([
                np.random.choice(pos_indices, min(pos_samples, len(pos_indices)), replace=True),
                np.random.choice(neg_indices, min(neg_samples, len(neg_indices)), replace=True),
                np.random.choice(neutral_indices, min(neutral_samples, len(neutral_indices)), replace=True)
            ])
        else:
            # If sample ratio is relatively balanced, perform normal random sampling
            indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.state[indices],
            self.action[indices],
            self.reward[indices],
            self.next_state[indices],
            self.done[indices]
        )
        
    def get_statistics(self):
        """Return buffer statistics"""
        return {
            'size': self.size,
            'positive_ratio': self.positive_experiences / self.size if self.size > 0 else 0,
            'negative_ratio': self.negative_experiences / self.size if self.size > 0 else 0,
            'collision_ratio': self.collision_count / self.size if self.size > 0 else 0
        }
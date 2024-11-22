import gymnasium as gym
import torch
from autopark_env.models.ddpg import Actor, Critic
from autopark_env.models.ddpg_net import DDPGNet
from autopark_env.models.replay_buffer import ReplayBuffer

# Monkey patch the reset method to remove the 'seed' argument
def patched_reset(self, **kwargs):
    kwargs.pop('seed', None)  # Remove 'seed' argument
    return self.original_reset(**kwargs)

# Apply the patch to the environment
gym.Env.original_reset = gym.Env.reset
gym.Env.reset = patched_reset

def test_models():
    try:
        # Create environment
        print("Creating environment...")
        env = gym.make('my-new-env-v0')
        
        # Get environment parameters
        print("\nGetting environment parameters...")
        state_dim = env.observation_space.shape[0]  # 6-dimensional state space
        action_dim = env.action_space.shape[0]      # 2-dimensional action space
        max_action = float(env.action_space.high[0])
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Max action value: {max_action}")
        
        # Initialize models
        print("\nInitializing models...")
        actor = Actor(state_dim, action_dim, max_action)
        critic = Critic(state_dim, action_dim)
        ddpg_net = DDPGNet(state_dim, action_dim, max_action)
        
        # Simple test
        print("\nTesting models with a sample state...")
        
        # Use the patched reset
        state, _ = env.reset()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        
        # Test Actor
        print("\nTesting Actor...")
        action = actor(state_tensor)
        print("Actor output shape:", action.shape)
        print("Actor output values:", action.detach().numpy())
        
        # Test Critic
        print("\nTesting Critic...")
        q_value = critic(state_tensor, action)
        print("Critic output shape:", q_value.shape)
        print("Critic output value:", q_value.detach().numpy())
        
        # Test DDPG-Net
        print("\nTesting DDPG-Net...")
        ddpg_action = ddpg_net(state_tensor)
        print("DDPG-Net output shape:", ddpg_action.shape)
        print("DDPG-Net output values:", ddpg_action.detach().numpy())
        
        # Test ReplayBuffer
        print("\nTesting ReplayBuffer...")
        buffer = ReplayBuffer(max_size=1000, state_dim=state_dim, action_size=action_dim)
        buffer.add(state, action.detach().numpy()[0], 0.0, state, False)
        print("Successfully added experience to buffer")
        
        print("\nAll tests completed successfully!")
        
    except gym.error.Error as gym_error:
        print(f"Gym error: {gym_error}")
    except ModuleNotFoundError as mod_error:
        print(f"Module not found error: {mod_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
    finally:
        # Ensure to call close only if env was created successfully
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    test_models()

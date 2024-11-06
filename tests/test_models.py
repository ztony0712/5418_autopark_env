import gymnasium as gym
import torch
from autopark_env.models.ddpg import Actor, Critic
from autopark_env.models.ddpg_net import DDPGNet
from autopark_env.models.replay_buffer import ReplayBuffer

# Monkey patch the reset method to remove the 'seed' argument
def patched_reset(self, **kwargs):
    kwargs.pop('seed', None)  # 移除 'seed' 参数
    return self.original_reset(**kwargs)

# Apply the patch to the environment
gym.Env.original_reset = gym.Env.reset
gym.Env.reset = patched_reset

def test_models():
    try:
        # 创建环境
        print("Creating environment...")
        env = gym.make('my-new-env-v0')
        
        # 获取环境参数
        print("\nGetting environment parameters...")
        state_dim = env.observation_space.shape[0]  # 6维状态空间
        action_dim = env.action_space.shape[0]      # 2维动作空间
        max_action = float(env.action_space.high[0])
        print(f"State dimension: {state_dim}")
        print(f"Action dimension: {action_dim}")
        print(f"Max action value: {max_action}")
        
        # 初始化模型
        print("\nInitializing models...")
        actor = Actor(state_dim, action_dim, max_action)
        critic = Critic(state_dim, action_dim)
        ddpg_net = DDPGNet(state_dim, action_dim, max_action)
        
        # 简单测试
        print("\nTesting models with a sample state...")
        
        # 使用patched后的reset
        state, _ = env.reset()
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # 添加batch维度
        
        # 测试Actor
        print("\nTesting Actor...")
        action = actor(state_tensor)
        print("Actor output shape:", action.shape)
        print("Actor output values:", action.detach().numpy())
        
        # 测试Critic
        print("\nTesting Critic...")
        q_value = critic(state_tensor, action)
        print("Critic output shape:", q_value.shape)
        print("Critic output value:", q_value.detach().numpy())
        
        # 测试DDPG-Net
        print("\nTesting DDPG-Net...")
        ddpg_action = ddpg_net(state_tensor)
        print("DDPG-Net output shape:", ddpg_action.shape)
        print("DDPG-Net output values:", ddpg_action.detach().numpy())
        
        # 测试ReplayBuffer
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
        # 确保只有在 env 被创建成功时才调用 close
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    test_models()

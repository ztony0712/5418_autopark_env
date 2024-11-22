# scripts/train.py

# # 训练DDPG
# python train.py --algo ddpg

# # 训练TD3
# python train.py --algo td3

# # 使用自定义配置文件
# python train.py --algo td3 --config configs/my_config.yaml

# # 禁用Tensorboard
# python train.py --algo ddpg --no-tensorboard

# # 设置随机种子
# python train.py --algo td3 --seed 42

import gymnasium as gym
import numpy as np
import torch
import argparse
from tensorboard import program
import threading
import webbrowser
import time
import os
import yaml

# 导入所有可能用到的模型和智能体
from autopark_env.models import ddpg, td3
from autopark_env.agents import ddpg_agent, td3_agent

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for various RL algorithms')
    
    # 基础参数
    parser.add_argument('--algo', type=str, default='ddpg', choices=['ddpg', 'td3'],
                      help='RL algorithm to use')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                      help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed')
    parser.add_argument('--no-tensorboard', action='store_true',
                      help='Disable tensorboard visualization')
    parser.add_argument('--save-dir', type=str, default='../models',
                      help='Directory to save models')
    
    return parser.parse_args()

def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def start_tensorboard():
    """Start Tensorboard"""
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'runs', '--bind_all'])
    url = tb.launch()
    print(f'TensorBoard running at {url}')
    webbrowser.open(url)

def create_agent(algo, config, env):
    """Create the corresponding agent based on the algorithm name"""
    state_dim = 12  # achieved_goal (6) + desired_goal (6)
    action_dim = 2  # [steering, acceleration]
    max_action = 1.0
    
    if algo == 'ddpg':
        actor = ddpg.Actor(state_dim, action_dim, max_action)
        critic = ddpg.Critic(state_dim, action_dim)
        return ddpg_agent.DDPGAgent(
            actor=actor,
            critic=critic,
            config=config,
            device=None
        )
    elif algo == 'td3':
        actor = td3.Actor(state_dim, action_dim, max_action)
        critic = td3.Critic(state_dim, action_dim)
        return td3_agent.TD3Agent(
            actor=actor,
            critic=critic,
            config=config,
            device=None
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

def train(args, config):
    """Main training function"""
    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
    # Start Tensorboard
    if not args.no_tensorboard:
        tb_thread = threading.Thread(target=start_tensorboard, daemon=True)
        tb_thread.start()
        time.sleep(2)
    
    # Create environment
    env = gym.make('my-new-env-v0')
    
    # Create agent
    agent = create_agent(args.algo, config, env)
    
    # Create save directory
    save_dir = os.path.join(args.save_dir, args.algo)
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        print(f"Starting {args.algo.upper()} training...")
        best_reward = float('-inf')
        episode_rewards = []
        reward_window = config['training']['reward_window']
        
        total_steps = 0
        total_episodes = int(config['training']['episodes'])
        
        for episode in range(total_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_start_time = time.time()
            
            for step in range(config['training']['max_steps']):
                total_steps += 1
                episode_steps += 1
                
                # Update exploration noise
                agent.update_exploration_noise(total_steps, total_episodes * config['training']['max_steps'])
                
                # Select action
                action = agent.select_action(state, add_noise=True)
                
                # Execute action
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store experience
                info['env'] = env
                agent.add_to_buffer(state, action, reward, next_state, done, info)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Process episode data
            agent.process_episode()
            
            # Calculate success rate and average reward
            success = info.get('is_success', False)
            success_rate = agent.update_success_rate(success)
            
            episode_rewards.append(episode_reward)
            if len(episode_rewards) > reward_window:
                episode_rewards.pop(0)
            avg_reward = np.mean(episode_rewards)
            
            # Log training metrics
            agent.writer.add_scalar('Training/Episode_Length', episode_steps, episode)
            agent.writer.add_scalar('Training/Episode_Reward', episode_reward, episode)
            agent.writer.add_scalar('Training/Average_Reward', avg_reward, episode)
            agent.writer.add_scalar('Training/Success_Rate', success_rate, episode)
            agent.writer.add_scalar('Training/Steps_Per_Second', 
                                  episode_steps / (time.time() - episode_start_time), episode)
            agent.writer.add_scalar('Training/Exploration_Noise', 
                                  agent.exploration_noise, 
                                  total_steps)
            
            # Train
            if len(agent.replay_buffer) > agent.batch_size:
                loss_info = agent.train()
                
                # Print training information
                if episode % config['training']['print_freq'] == 0:
                    print(f"Episode {episode}")
                    print(f"  Reward: {episode_reward:.2f} (Avg: {avg_reward:.2f})")
                    print(f"  Success Rate: {success_rate:.2%}")
                    print(f"  Steps: {episode_steps}")
                    if loss_info is not None:
                        if 'actor_loss' in loss_info and loss_info['actor_loss'] is not None:
                            print(f"  Actor Loss: {loss_info['actor_loss']:.4f}")
                        print(f"  Critic Loss: {loss_info['critic_loss']:.4f}")
                        print(f"  Q-value Mean: {loss_info['q_value_mean']:.4f}")
                        if 'q_diff_mean' in loss_info:
                            print(f"  Q-networks Difference: {loss_info['q_diff_mean']:.4f}")
                    print(f"  Buffer Size: {len(agent.replay_buffer)}")
                    print("----------------------------------------")
            
            # Save model
            if avg_reward > best_reward:
                best_reward = avg_reward
                agent.save(f"{save_dir}/best_model.pth")
            
            if episode % config['training']['save_freq'] == 0:
                agent.save(f"{save_dir}/checkpoint_{episode}.pth")
                
        # Save final model
        agent.save(f"{save_dir}/final_model.pth")
        print(f"Training completed. Final model saved to {save_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save(f"{save_dir}/interrupted_model.pth")
        print(f"Model saved to {save_dir}")
    
    finally:
        env.close()

def main():
    args = parse_args()
    config = load_config(args.config)
    train(args, config)

if __name__ == "__main__":
    main()

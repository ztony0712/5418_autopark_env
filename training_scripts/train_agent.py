# scripts/train_agent.py
import autopark_env  # 添加环境导入
import gymnasium as gym
from autopark_env.agents.parking_agent import ParkingAgent  # 我们将使用新的TD3Agent
import torch
from tensorboard import program
import threading
import webbrowser
import time

# 训练参数
LEARNING_EPISODES = int(1e4)
MODEL_SAVE_PATH = 'saved_models'
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
GAMMA = 0.99
MAX_TIMESTEPS = 1000

# Tensorboard配置
def start_tensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'runs', '--bind_all'])
    url = tb.launch()
    print(f'TensorBoard running at {url}')
    webbrowser.open(url)

def main():
    # 启动Tensorboard
    tb_thread = threading.Thread(target=start_tensorboard, daemon=True)
    tb_thread.start()
    time.sleep(2)  # 等待Tensorboard启动
    
    # 创建环境和智能体
    env = gym.make('my-new-env-v0')
    
    agent = ParkingAgent(
        env_name='my-new-env-v0',
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        max_timesteps=MAX_TIMESTEPS,
        batch_size=BATCH_SIZE
    )
    
    try:
        # 训练
        print("Starting training...")
        agent.train(num_episodes=LEARNING_EPISODES, save_interval=1000, test_env=env)
        
        # 保存模型
        agent.save_model(MODEL_SAVE_PATH)
        print(f"Training completed. Model saved to {MODEL_SAVE_PATH}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save_model(MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    
    finally:
        env.close()

if __name__ == "__main__":
    main()

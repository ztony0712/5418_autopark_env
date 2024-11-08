# scripts/train_agent.py
import os
from autopark_env.agents.parking_agent import ParkingAgent

def main():
    # 实例化 ParkingAgent
    agent = ParkingAgent(env_name='my-new-env-v0')
    
    # 检查模型文件是否存在，如果存在则加载模型
    if os.path.exists('saved_models/actor.pth') and os.path.exists('saved_models/critic.pth'):
        agent.load_model(path='saved_models/')
    else:
        print("No saved model found. Starting training from scratch.")
    
    # 开始训练
    agent.train(num_episodes=100)  # 训练 50000
    
    # 保存训练好的模型
    agent.save_model(path='saved_models/')
    
    # 测试训练好的模型
    print("Starting testing...")
    agent.test(num_episodes=5)  # 测试 5 个回合

if __name__ == "__main__":
    main()

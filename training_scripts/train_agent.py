# scripts/train_agent.py
import autopark_env  # 添加环境导入
import gymnasium as gym
from autopark_env.agents.parking_agent import ParkingAgent
import threading
import time
from tensorboard import program
import webbrowser
import os

LEARNING_EPISODES = int(1e5)  # 训练的总回合数
MODEL_SAVE_PATH = 'saved_models'  # 模型保存路径

# 启动 TensorBoard 在后台运行
def start_tensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'runs', '--bind_all'])  # 添加 bind_all 使其可以从其他设备访问
    url = tb.launch()
    print(f'TensorBoard is running at {url}')
    webbrowser.open(url)  # 自动在浏览器中打开 TensorBoard
    while True:
        time.sleep(1)  # 保持 TensorBoard 运行

# 在后台线程启动 TensorBoard
tb_thread = threading.Thread(target=start_tensorboard, daemon=True)
tb_thread.start()

# 等待 TensorBoard 完全启动
time.sleep(3)

def main():
    # 创建环境
    env = gym.make('my-new-env-v0')
    obs, info = env.reset()  # 添加环境重置
    
    # 创建停车代理
    agent = ParkingAgent(env_name='my-new-env-v0', learning_rate=0.001, gamma=0.99, max_timesteps=1000)

    # 检查是否存在已保存的模型，并加载
    if os.path.exists(MODEL_SAVE_PATH):
        agent.load_model(MODEL_SAVE_PATH)
        print(f"Loaded model from {MODEL_SAVE_PATH}")

    # 开始训练
    agent.train(num_episodes=LEARNING_EPISODES)

    # 训练完成后保存模型
    agent.save_model(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()

# 等待用户手动关闭
print("\nTraining finished. Press Ctrl+C to close TensorBoard and exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting...")

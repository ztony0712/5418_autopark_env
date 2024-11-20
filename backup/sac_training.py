# test_env.py
import autopark_env  # Ensure autopark_env is correctly imported
import gymnasium as gym
from autopark_env.envs.my_new_env import MyNewEnv  # Import our defined parking environment
from stable_baselines3 import HerReplayBuffer, SAC
from tensorboard import program
import threading
import time
import webbrowser
import torch  # 添加这行
import os     # 添加这行

LEARNING_STEPS = 3e5

# 启动 TensorBoard 在后台运行
def start_tensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'logs', '--bind_all'])  # 添加 bind_all 使其可以从其他设备访问
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

# 创建环境和模型
env = gym.make('my-new-env-v0')
obs, info = env.reset()

her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')

# 在创建模型之前，设置 CUDA 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 可以设置要使用的 GPU 编号（如果有多个 GPU）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用第一个 GPU

# 创建模型时添加 device 参数
model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs, verbose=1, 
            tensorboard_log="logs", 
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=2048, tau=0.05,
            policy_kwargs=dict(net_arch=[1024, 1024, 1024]),
            learning_starts=10000,
            device=device)
    
model.learn(int(LEARNING_STEPS))
model.save("models/sac_parking_final")

env.close()

# 等待用户手动关闭
print("\nTraining finished. Press Ctrl+C to close TensorBoard and exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting...")

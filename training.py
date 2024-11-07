# test_env.py
import autopark_env  # Ensure autopark_env is correctly imported
import gymnasium as gym
from autopark_env.envs.my_new_env import MyNewEnv  # Import our defined parking environment
from stable_baselines3 import HerReplayBuffer, SAC
from tensorboard import program
import threading
import time
import webbrowser

LEARNING_STEPS = 5e4

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
env = MyNewEnv()
obs, info = env.reset()

her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')
model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=her_kwargs, verbose=1, 
            tensorboard_log="logs", 
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95, batch_size=1024, tau=0.05,
            policy_kwargs=dict(net_arch=[512, 512, 512]),
            learning_starts=10000)
    
model.learn(int(LEARNING_STEPS))

env.close()

# 等待用户手动关闭
print("\nTraining finished. Press Ctrl+C to close TensorBoard and exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting...")

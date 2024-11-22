# test_env.py
import autopark_env  # Ensure autopark_env is correctly imported
import gymnasium as gym
from autopark_env.envs.my_new_env import MyNewEnv  # Import our defined parking environment
from stable_baselines3 import HerReplayBuffer, SAC
from tensorboard import program
import threading
import time
import webbrowser
import torch  # Add this line
import os     # Add this line

LEARNING_STEPS = 1e2

# Start TensorBoard in the background
def start_tensorboard():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', 'logs', '--bind_all'])  # Add bind_all to allow access from other devices
    url = tb.launch()
    print(f'TensorBoard is running at {url}')
    webbrowser.open(url)  # Automatically open TensorBoard in the browser
    while True:
        time.sleep(1)  # Keep TensorBoard running

# Start TensorBoard in a background thread
tb_thread = threading.Thread(target=start_tensorboard, daemon=True)
tb_thread.start()

# Wait for TensorBoard to fully start
time.sleep(3)

# Create the environment and model
env = gym.make('my-new-env-v0')
obs, info = env.reset()

her_kwargs = dict(n_sampled_goal=4, goal_selection_strategy='future')

# Set CUDA device before creating the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set the GPU number to use (if multiple GPUs are available)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU

# Create the model with the device parameter
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
model.save("../models/mpc_sac_parking_final")

env.close()

# Wait for the user to manually close
print("\nTraining finished. Press Ctrl+C to close TensorBoard and exit.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting...")

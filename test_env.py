# test_env.py
import autopark_env  # Ensure autopark_env is correctly imported
import gymnasium as gym
from autopark_env.envs.my_new_env import MyNewEnv  # Import our defined parking environment

# Use MyNewEnv for testing
env = MyNewEnv()  # Create an instance of the environment
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Randomly select an action
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # Display vehicle position
env.close()

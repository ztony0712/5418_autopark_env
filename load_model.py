import autopark_env  # Ensure autopark_env is correctly imported
from autopark_env.envs.my_new_env import MyNewEnv  # Import our defined parking environment
from stable_baselines3 import SAC
from tqdm import trange
import gymnasium as gym

env = gym.make('my-new-env-v0')
obs, info = env.reset()
done = False

model = SAC.load("sac_autopark_model", env=env)

N_EPISODES = 10  # @param {type: "integer"}

for episode in trange(N_EPISODES, desc="Test episodes"):
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()

env.close()
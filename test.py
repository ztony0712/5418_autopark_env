import gymnasium
import highway_env
from matplotlib import pyplot as plt

env = gymnasium.make('highway-v0', render_mode='rgb_array')
env.reset()
for _ in range(30):
    action = env.unwrapped.action_type.actions_indexes["IDLE"]
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()

# /home/nuplan/.local/lib/python3.8/site-packages/highway_env/envs/sac_net.py
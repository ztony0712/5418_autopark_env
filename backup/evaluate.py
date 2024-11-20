import autopark_env  # Ensure autopark_env is correctly imported
from autopark_env.envs.my_new_env import MyNewEnv  # Import our defined parking environment
from stable_baselines3 import SAC
from tqdm import trange
import gymnasium as gym
import numpy as np

env = gym.make('my-new-env-v0', render_mode="human")
obs, info = env.reset()
done = False

model = SAC.load("../models/sac_parking_final", env=env)

N_EPISODES = 50  # @param {type: "integer"}
success_count = 0  # 成功泊车的次数
total_parking_time = 0  # 总泊车时间
trajectories = []  # 存储轨迹
smoothness_scores = []  # 存储每个回合的平滑性分数

for episode in trange(N_EPISODES, desc="Test episodes"):
    obs, info = env.reset()
    done = truncated = False
    episode_trajectory = []  # 当前回合的轨迹
    episode_time = 0  # 当前回合的泊车时间

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        episode_trajectory.append(obs['observation'])  # 记录轨迹，提取'observation'
        episode_time += 1  # 增加泊车时间

    # 统计成功率
    if done and info.get('is_success', False):  # 假设info中有'success'字段
        success_count += 1

    total_parking_time += episode_time
    trajectories.append(episode_trajectory)  # 存储当前回合轨迹

    # 计算轨迹平滑性
    if len(episode_trajectory) > 1:
        distances = []
        for i in range(1, len(episode_trajectory)):
            dist = np.linalg.norm(np.array(episode_trajectory[i]) - np.array(episode_trajectory[i - 1]))
            distances.append(dist)
        smoothness_score = np.mean(distances)  # 计算平均距离作为平滑性分数
        smoothness_scores.append(smoothness_score)

# 计算成功率和平均泊车时间
success_rate = success_count / N_EPISODES
average_parking_time = total_parking_time / N_EPISODES
average_smoothness = np.mean(smoothness_scores) if smoothness_scores else 0  # 计算平均平滑性

print(f"成功率: {success_rate:.2f}")
print(f"平均泊车时间: {average_parking_time:.2f}步")
print(f"平均轨迹平滑性: {average_smoothness:.2f}")

env.close()
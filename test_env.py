# test_env.py
import highway_env  # 确保 highway_env 正确导入
import gymnasium as gym
from highway_env.envs.my_new_env import MyNewEnv  # 导入我们定义的停车环境

# 使用 MyNewEnv 进行测试
env = MyNewEnv()  # 创建环境实例
obs, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # 随机选择一个动作
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # 显示车辆位置
env.close()




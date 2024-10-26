import gym
from highway_env.envs.my_new_env import MyNewEnv
from .agent import DDPGAgent

def train_ddpg_agent(env, agent, num_episodes=1000, batch_size=64):
    for episode in range(num_episodes):
        state, _ = env.reset()  # 重置环境
        done = False

        while not done:
            action = agent.select_action(state)  # 选择动作
            next_state, reward, done, _, _ = env.step(action)  # 执行动作
            agent.replay_buffer.add((state, action, reward, next_state, done))  # 存储经验
            agent.train(batch_size)  # 训练代理

            state = next_state

if __name__ == "__main__":
    env = MyNewEnv()  # 实例化你的环境
    agent = DDPGAgent(state_dim=6, action_dim=2)  # 根据你的环境设置
    train_ddpg_agent(env, agent)

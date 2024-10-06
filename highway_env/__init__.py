# highway_env/__init__.py

from gymnasium.envs.registration import register

def register_my_envs():
    # 注册你的自定义环境
    register(
        id='my-new-env-v0',
        entry_point='highway_env.envs.my_new_env:MyNewEnv',
        max_episode_steps=1000,
        reward_threshold=100.0,
    )

# 执行注册
register_my_envs()


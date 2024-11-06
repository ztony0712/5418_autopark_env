# autopark_env/__init__.py

from gymnasium.envs.registration import register

def register_my_envs():
    # Register custom environment
    register(
    	id='my-new-env-v0',
    	entry_point='autopark_env.envs.my_new_env:MyNewEnv',
        max_episode_steps=1000,
        reward_threshold=100.0,
    )

# Execute registration
register_my_envs()

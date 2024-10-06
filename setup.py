from setuptools import setup, find_packages

setup(
    name='highway_env',
    version='0.1',
    packages=find_packages(),
    install_requires=['gymnasium', 'numpy'],
    entry_points={
        'gym.envs': [
            'my-new-env-v0 = highway_env.envs.my_new_env:MyNewEnv',
        ],
    },
)



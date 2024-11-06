from setuptools import setup, find_packages

setup(
    name='autopark_env',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['gymnasium', 'numpy', 'pygame'],
    entry_points={
        'gym.envs': [
            'my-new-env-v0 = autopark_env.envs.my_new_env:MyNewEnv',
        ],
    },
)



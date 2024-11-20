import os
import base64
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from IPython import display as ipythondisplay
import numpy as np
from pyvirtualdisplay import Display
import torch
import gymnasium as gym

from autopark_env.agents.parking_agent import ParkingAgent

display = Display(visible=0, size=(1400, 900))
display.start()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def record_videos(env, video_folder="videos"):
    wrapped = RecordVideo(
        env, video_folder=video_folder, episode_trigger=lambda e: True
    )

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped


def show_videos(path="videos"):
    html = []
    for mp4 in Path(path).glob("*.mp4"):
        video_b64 = base64.b64encode(mp4.read_bytes())
        html.append(
            """<video alt="{}" autoplay
                      loop controls style="height: 400px;">
                      <source src="data:video/mp4;base64,{}" type="video/mp4" />
                 </video>""".format(
                mp4, video_b64.decode("ascii")
            )
        )
    ipythondisplay.display(ipythondisplay.HTML(data="<br>".join(html)))


env = gym.make('my-new-env-v0', render_mode="rgb_array")
obs, info = env.reset()
done = truncated = False

agent = ParkingAgent(env_name='my-new-env-v0')
agent.load_model(path='training_scripts/saved_models')

N_EPISODES = 10  # @param {type: "integer"}

# Enable video recording
env = record_videos(env)

for episode in range(N_EPISODES):
    state, _ = env.reset()
    state = state['observation']
    # Reset the agent's environment
    agent.env = env  # Ensure the agent uses the same environment
    episode_reward = 0
    done = truncated = False
    
    while not (done or truncated):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # 在使用模型之前，将state_tensor移到正确的设备上
        state_tensor = state_tensor.to(device)  # device应该在之前定义为 cuda 或 cpu
        
        with torch.no_grad():
            action = agent.actor(state_tensor).cpu().numpy()[0]
        
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(action)  # Use the main environment instead of agent.env
        done = terminated or truncated
        state = next_state['observation']
        episode_reward += reward
        env.render()  # Render on the main environment
        
    print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

env.close()
show_videos()  # Uncomment to display videos
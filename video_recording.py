import base64
from pathlib import Path

from gymnasium.wrappers import RecordVideo
from IPython import display as ipythondisplay
import numpy as np
from pyvirtualdisplay import Display
import torch
from tqdm import trange
import gymnasium as gym

from autopark_env.agents.parking_agent import ParkingAgent

display = Display(visible=0, size=(1400, 900))
display.start()


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
agent.load_model(path='saved_models/')

N_EPISODES = 10  # @param {type: "integer"}

# 启用视频录制
env = record_videos(env)

for episode in range(N_EPISODES):
    state, _ = env.reset()
    state = state['observation']
    # 重置agent的环境
    agent.env = env  # 确保agent使用相同的环境
    episode_reward = 0
    done = truncated = False
    
    while not (done or truncated):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = agent.actor(state_tensor).cpu().numpy()[0]
        
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_state, reward, terminated, truncated, _ = env.step(action)  # 使用主环境而不是agent.env
        done = terminated or truncated
        state = next_state['observation']
        episode_reward += reward
        env.render()  # 在主环境上渲染
        
    print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

env.close()
show_videos()  # 取消注释以显示视频
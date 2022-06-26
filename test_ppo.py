from gym3 import VideoRecorderWrapper
import torch
from procgen import ProcgenEnv
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder
from utils.wrappers import VecPyTorch, VecExtractDictObs, VecMonitor
from utils.agent import Agent

import time
import numpy as np
import random
import os

exp_name = os.path.basename(__file__).rstrip(".py")
seed = 1
torch_deterministic = True
num_envs = 1
cuda = True
gym_id = "ebigfishs"
experiment_name = f"{gym_id}__{exp_name}__{seed}__{int(time.time())}"


device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic
venv = ProcgenEnv(num_envs=num_envs, env_name=gym_id, num_levels=0, start_level=0, distribution_mode='hard')
venv = VecExtractDictObs(venv, "rgb")
venv = VecMonitor(venv=venv)
envs = VecNormalize(venv=venv, norm_obs=False)
envs = VecPyTorch(envs, device)
envs = VecVideoRecorder(envs, f'videos/{experiment_name}', record_video_trigger=lambda x: x == 0, video_length=6000,)

agent = Agent(envs=envs).to(device=device)
agent.load_state_dict(torch.load("models/train_ppo"))
agent.eval()

episodes = 5
for i in range(episodes):
    obs = envs.reset()
    total_reward = 0
    done = False
    while not done:
        envs.render(mode="rgb_array")
        action, _, _ = agent.get_action(obs)
        obs, reward, done, info = envs.step(action)
        total_reward += reward

    print(f"Total Reward: {total_reward}")

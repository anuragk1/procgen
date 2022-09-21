import torch
from procgen import ProcgenEnv
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder, VecFrameStack, VecExtractDictObs, VecMonitor
from utils.wrappers import VecPyTorch#, VecExtractDictObs, VecMonitor
from utils.agent import MLPAgent, CNNAgent, AttAgent
from utils.policies import preprocess_obs

import time
import numpy as np
import random
import os

exp_name = os.path.basename(__file__).rstrip(".py")
seed = 1
torch_deterministic = True
num_envs = 1
cuda = True
gym_id = "bigfishr"
experiment_name = f"{gym_id}__{exp_name}__{seed}__{int(time.time())}"


device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = torch_deterministic
venv = ProcgenEnv(num_envs=num_envs, env_name=gym_id, num_levels=0, start_level=0, distribution_mode='hard')
venv = VecExtractDictObs(venv, "positions")
venv = VecFrameStack(venv, n_stack=2)
venv = VecMonitor(venv=venv)
envs = VecNormalize(venv=venv, norm_obs=False)
envs = VecPyTorch(venv, device)
envs = VecVideoRecorder(envs, f'videos/{experiment_name}', record_video_trigger=lambda x: x == 0, video_length=1000,)

agent = AttAgent(envs=envs).to(device=device)
agent.load_state_dict(torch.load("models/train_att"))
agent.eval()

eat_env = ProcgenEnv(num_envs=1, env_name="ebigfishs", num_levels=0, start_level=0, distribution_mode='hard')
eat_env = VecExtractDictObs(eat_env, "positions")
eat_env = VecFrameStack(eat_env, n_stack=2)
eat_env = VecMonitor(venv=eat_env) 
eat_env = VecNormalize(venv=eat_env, norm_obs=False)
eat_env = VecPyTorch(eat_env, device)

agent_eat = MLPAgent(envs=eat_env).to(device=device)
agent_eat.load_state_dict(torch.load("models/train_ppo2_100M"))
agent_eat.eval()

dodge_env = ProcgenEnv(num_envs=1, env_name="ebigfishl", num_levels=0, start_level=0, distribution_mode='hard')
dodge_env = VecExtractDictObs(dodge_env, "positions")
dodge_env = VecFrameStack(dodge_env, n_stack=2)
dodge_env = VecMonitor(venv=dodge_env) 
dodge_env = VecNormalize(venv=dodge_env, norm_obs=False)
dodge_env = VecPyTorch(dodge_env, device)

agent_dodge = MLPAgent(envs=dodge_env).to(device=device)
agent_dodge.load_state_dict(torch.load("models/train_ppo2_dodgefish_1B"))
agent_dodge.eval()

episodes = 5
for i in range(episodes):
    obs = envs.reset()
    obs = obs.view(1, np.array(envs.observation_space.shape).prod())
    total_reward = 0
    done = False
    while not done:
        envs.render(mode="rgb_array")
        action, _, _ = agent.get_action(obs)
        obs = preprocess_obs(obs)
        if action == 0:
            action_, _, _ = agent_eat.get_action(obs)
        if action == 1:
            action_, _, _ = agent_dodge.get_action(obs)
        action = action_
        obs, reward, done, info = envs.step(action)
        obs = obs.view(1, np.array(envs.observation_space.shape).prod())
        total_reward += reward

    print(f"Total Reward: {total_reward}")


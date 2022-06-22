from statistics import mode
from procgen import ProcgenEnv, ProcgenGym3Env
import numpy as np
import os

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import VecExtractDictObs, VecMonitor

models_dir = "models/PPO_SingleFish_2/"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env_name = "ebigfishs"

env = ProcgenEnv(num_envs=32, env_name=env_name)
env = VecMonitor(venv=env)

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, device='cuda')

TIMESTEPS = 2000000
model.learn(total_timesteps=TIMESTEPS, tb_log_name="PPO_2")
model.save(models_dir)
# iters = 0
# for i in range(50):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
#     model.save(f"{models_dir}/{TIMESTEPS*i}")
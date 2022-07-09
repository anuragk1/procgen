from procgen import ProcgenEnv, ProcgenGym3Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
import gym3
import numpy as np

env_name = "ebigfishs"

env = ProcgenEnv(num_envs=1, env_name=env_name, render_mode="rgb_array")
# env = VecMonitor(venv=env, filename=None,)
# env = VecVideoRecorder(venv=env, video_folder="./", record_video_trigger=lambda x: x == 0, video_length=5000, name_prefix='PPO_SingleFish_2')
env = VecVideoRecorder(venv=env, video_folder="./", record_video_trigger=lambda x: x == 0, video_length=1500, name_prefix='PPO_MultipleFish_0.5')
# model = PPO.load("models/PPO_SingleFish_2/PPO_SingleFish_2.zip", env=env)
# model = PPO.load("models/PPO_MultipleFish_0.5/PPO_MultipleFish_0.5_2", env=env)

print(env.env.env.combos)
# episodes = 2
# for i in range(episodes):
#     obs = env.reset()
#     total_reward = 0
#     done = False

#     while not done:
#         env.render(mode="rgb_array")
#         # action, _ = model.predict(observation=obs, deterministic=True)
#         action = [env.action_space.sample()]
#         action = np.array(action)
#         obs, reward, done, info = env.step(action)
#         total_reward += reward

#     print(f"Total Reward: {total_reward}")

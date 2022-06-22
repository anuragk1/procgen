from procgen import ProcgenEnv, ProcgenGym3Env
from stable_baselines3 import PPO
import gym3
import numpy as np

env_name = "ebigfishs"

# class RenderingEnv(gym3.ViewerWrapper):
#     def __init__(self, env_name):
#         env = ProcgenGym3Env(num=1, env_name=env_name, render_mode="rgb_array")
#         super().__init__(env, info_key="rgb")

#     def render(self):
#         self.env.render()

# env = RenderingEnv(env_name=env_name)

env = ProcgenEnv(num_envs=1, env_name=env_name, render_mode="rgb_array")
model = PPO.load("models/PPO_SingleFish_2/PPO_SingleFish_2.zip", env=env)

episodes = 5
for i in range(episodes):
    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        env.render(mode="rgb_array")
        action, _ = model.predict(observation=obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Total Reward: {total_reward}")

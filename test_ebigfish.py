from utils_procgen import InteractiveEnv
from procgen import ProcgenGym3Env
import numpy as np
import sys

INTERACTIVE = True

env_name = "bigfishr"
obj_list = ["fish_pos", "fish_count", "fish_id"]

if INTERACTIVE:
    env = InteractiveEnv(env_name)
else:
    env = ProcgenGym3Env(num=1, env_name=env_name)

step = 1
NB_DONE = 0
TO_SUCCEED = 1
# env_info = env.get_info()[0]
# print(env_info.keys())

total_reward = 0
fish_dict = {}

while NB_DONE < TO_SUCCEED:
    if INTERACTIVE:
        env.iact()  # Slow because renders
    else:
        env.act(np.random.randint((env.ac_space.eltype.n,)))
    
    rew, obs, done = env.observe()
    print(f"rew : {rew}")
    # print(f"observation : {obs['positions']}")
    # total_reward += rew
    # all_objects = env.get_info()[0]
    # agent_pos = all_objects["agent_pos"]
    # fish_ids = all_objects["fish_id"]
    # fish_pos = all_objects["fish_pos"]
    
    # for id, pos in zip(fish_ids, fish_pos):
    #     if id != 0 and id not in fish_dict.keys():
    #         # print(f"fish id: {id}, fish_pos: {pos}")
    #         fish_dict.update({id: pos})
        
    #     fish_dict[id] = pos
    
    # for id in list(fish_dict.keys()):
    #     if id not in fish_ids or id == 0:
    #         fish_dict.pop(id)

    # print(f"Agent_pos: {agent_pos}")
    # for id, pos in fish_dict.items():
    #     print(f"id: {id}        position: {pos}")
    # print(f"Total Reward: {total_reward}")
    # print('-'*20)

    # if done:
    #     print(f"Done in {step} steps")
    #     step = 0
    #     NB_DONE += 1
    #     total_reward = 0
    #     fish_dict.clear()
    # step += 1
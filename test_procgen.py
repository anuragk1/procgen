from utils_procgen import InteractiveEnv
import numpy as np
import sys


env_name = "ecoinrun"
env = InteractiveEnv(env_name)
step = 1
NB_DONE = 0
TO_SUCCEED = 3
frames = []
while NB_DONE < TO_SUCCEED:
    env.iact()
    rew, obs, first = env.observe()
    all_objects = env.get_info()[0]
    agent_pos = all_objects["agent_pos"]
    print(f"agent pos: {agent_pos}")
    for obj in ["coin_pos", "saw1_pos", "saw2_pos", "saw3_pos", "saw4_pos"]:
        obj_pos = all_objects[obj]
        if obj_pos[0] > 0 and np.abs(agent_pos[0] - obj_pos[0]) < 7:
            print(f"{obj}: ", all_objects[obj])
    print('-'*20)
    if rew != 0:
        print(f"Done in {step} steps")
        step = 0
        NB_DONE += 1
    step += 1

from utils_procgen import InteractiveEnv
from procgen import ProcgenGym3Env
import numpy as np
import sys

INTERACTIVE = True

# env_name = "ecoinrun"
# obj_list = ["coin_pos", "saw1_pos", "saw2_pos", "saw3_pos", "saw4_pos"]
env_name = "edodgeball"
obj_list = ["door_pos", "enemy1_pos", "enemy2_pos", "enemy3_pos", "enemy4_pos"]
if INTERACTIVE:
    env = InteractiveEnv(env_name)
else:
    env = ProcgenGym3Env(num=1, env_name=env_name)
step = 1
NB_DONE = 0
TO_SUCCEED = 3
frames = []
while NB_DONE < TO_SUCCEED:
    if INTERACTIVE:
        env.iact()  # Slow because renders
    else:
        env.act(np.random.randint((env.ac_space.eltype.n,)))
    rew, obs, done = env.observe()
    all_objects = env.get_info()[0]
    if env_name[0] == "e":
        agent_pos = all_objects["agent_pos"]
        print(f"agent pos: {agent_pos}")
    for obj in obj_list:
        obj_pos = all_objects[obj]
        if env_name == "edodgeball":
            print(f"{obj}: ", all_objects[obj])
        elif env_name == "ecoinrun":
            if obj_pos[0] > 0 and np.abs(agent_pos[0] - obj_pos[0]) < 7:
                print(f"{obj}: ", all_objects[obj])
    print('-'*20)
    if done:
        print(f"Done in {step} steps")
        step = 0
        NB_DONE += 1
    step += 1

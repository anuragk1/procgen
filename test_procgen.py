from utils_procgen import InteractiveEnv
from procgen import ProcgenGym3Env
import numpy as np
import sys

INTERACTIVE = True

# env_name = "ecoinrun"
# obj_list = ["coin_pos", "saws_pos"]
# env_name = "edodgeball"
# obj_list = ["door_pos", "enemy1_pos", "enemy2_pos", "enemy3_pos", "enemy4_pos"]
env_name = "ebigfish"
obj_list = ["fish_pos", "fish_count"]

def print_obj(env_name, obj_pos, obj_name="Unknown", agent_pos=None):
    if env_name == "edodgeball":
        print(f"{obj_name}: ", obj_pos)
    elif env_name == "ecoinrun":
        if obj_pos[0] > 0 and np.abs(agent_pos[0] - obj_pos[0]) < 7:
            print(f"{obj_name}: ", obj_pos)
    elif env_name == "ebigfish":
        print(f"{obj_name}: ", obj_pos)

if INTERACTIVE:
    env = InteractiveEnv(env_name)
else:
    env = ProcgenGym3Env(num=1, env_name=env_name)
step = 1
NB_DONE = 0
TO_SUCCEED = 3
frames = []
# env_info = env.get_info()[0]
# print(env_info.keys())
while NB_DONE < TO_SUCCEED:
    if INTERACTIVE:
        env.iact()  # Slow because renders
    else:
        env.act(np.random.randint((env.ac_space.eltype.n,)))
    rew, obs, done = env.observe()
    all_objects = env.get_info()[0]
    agent_pos = all_objects["agent_pos"]
    # print(agent_pos)
    for name, obj_pos in all_objects.items():
        if name in ["prev_level_seed", "prev_level_complete", "level_seed", "rgb"]:
            continue
        if obj_pos.ndim == 2:
            # if step > 100:
            #     import ipdb; ipdb.set_trace()
            for i, obj_p in enumerate(obj_pos):
                print_obj(env_name, obj_p, f"{name} [{i}]", agent_pos)
        else:
            print_obj(env_name, obj_pos, name, agent_pos)

    print('-'*20)
    if done:
        print(f"Done in {step} steps")
        step = 0
        NB_DONE += 1
    step += 1

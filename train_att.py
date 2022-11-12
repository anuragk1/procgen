import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from procgen import ProcgenEnv
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder, VecFrameStack, VecExtractDictObs, VecMonitor
from utils.wrappers import VecPyTorch#, VecExtractDictObs, VecMonitor
from utils.agent import CNNAgent, MLPAgent, AttAgent
from utils.policies import preprocess_obs, preprocess_obs_r, split_obs

torch.autograd.set_detect_anomaly(True)

exp_name = os.path.basename(__file__).rstrip(".py")
gym_id = "bigfishr"
learning_rate = 2.5e-4
seed = 1
total_timesteps = int(1e7)
torch_deterministic = True
cuda = True
prod_mode = False
wandb_proj_name = "AttRL"
wandb_entity = None
capture_video = False
save_path = f"models/{exp_name}_{gym_id}"

num_minibatches = 16
num_envs = 256*4
num_steps = 256
gamma = 0.95
gae_lambda = 0.95
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
clip_coef = 0.2
update_epochs = 3
kle_stop = True
kle_rollback = True
target_kl = 0.03
gae = True
norm_adv = True
anneal_lr = True
clip_vloss = True

if not seed:
    seed = int(time.time())

batch_size = int(num_envs * num_steps)
minibatch_size = int(batch_size // num_minibatches)

experiment_name = f"{gym_id}__{exp_name}__{seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")

if prod_mode:
    import wandb
    wandb.init(project=wandb_proj_name, entity=wandb_entity, sync_tensorboard=True, name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"runs/{experiment_name}")


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
# envs = VecPyTorch(envs, device)

agent = AttAgent(envs=envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
if anneal_lr:
    lr = lambda f: f * learning_rate

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
agent_dodge.load_state_dict(torch.load("models/train_ppo2_dodgefish_1B_with_better_reward_function"))
agent_dodge.eval()

obs_space_flat = np.array(envs.observation_space.shape).prod()

obs = torch.zeros((num_steps, num_envs, obs_space_flat)).to(device)
processed_obs = torch.zeros((num_steps, num_envs, 8)).to(device)
processed_obs_r = torch.zeros((num_steps, num_envs, 4)).to(device)
actions = torch.zeros((num_steps, num_envs) + envs.action_space.shape).to(device)
logprobs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
# info_id_comps = torch.zeros((num_steps, num_envs), dtype=torch.bool).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)

global_step = 0
start_time = time.time()

next_obs = envs.reset()
next_obs = next_obs.view(num_envs, np.array(envs.observation_space.shape).prod())
next_done = torch.zeros(num_envs).to(device)
num_updates = int(total_timesteps // batch_size)

# present_info_id = torch.zeros(num_envs, dtype=torch.int64).to(device)
# old_info_id = torch.zeros(num_envs, dtype=torch.int64).to(device)
action = torch.zeros(num_envs, dtype=torch.int64).to(device)
logproba = torch.zeros(num_envs,).to(device)

# fish_rewards = []

for update in range(1, num_updates+1):
    if anneal_lr:
        frac = 1.0 - (update - 1.0) / (num_updates*8)
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow
        # info_id_comp = old_info_id != present_info_id

    for step in range(0, num_steps):
        global_step += 1 * num_envs
        obs[step] = next_obs
        dones[step] = next_done

        processed_obs[step] = preprocess_obs(obs[step])
        processed_obs_r[step] = preprocess_obs_r(obs[step])

        # info_id_comps[step] = info_id_comp

        # with torch.no_grad():
        #     if torch.numel(info_id_comp[info_id_comp == True]):
        #         values[step][info_id_comp] = agent.get_value(processed_obs_r[step][info_id_comp]).flatten()
        #         action[info_id_comp], logproba[info_id_comp], _ = agent.get_action(processed_obs_r[step][info_id_comp])

        with torch.no_grad():
            values[step] = agent.get_value(processed_obs_r[step]).flatten()
            action, logproba, _ = agent.get_action(processed_obs_r[step])

        actions[step] = action
        expert_actions = torch.zeros_like(action).to(device)
        logprobs[step] = logproba

        # a0_and_id_comp = torch.logical_and(action == 0, info_id_comp)
        # a1_and_id_comp = torch.logical_and(action == 1, info_id_comp)

        # print(f"action == 0 : {action == 0}")
        # print(f"info_id_comp : { torch.numel(info_id_comp[info_id_comp == True])}")
        # print(f"a0_and_id_comp : {a0_and_id_comp}")
        # print(f"a1_and_id_comp : {a1_and_id_comp}")
        
        if torch.numel(action[action == 0]) != 0:  
            expert_actions[action == 0] = agent_eat.get_action(processed_obs[step][action == 0])[0]

        if torch.numel(action[action == 1]) != 0:
            expert_actions[action == 1] = agent_dodge.get_action(processed_obs[step][action == 1])[0]

        # old_info_id = present_info_id
        next_obs, rs, ds, infos = envs.step(expert_actions)
        # ids = np.array([info_dict['fish_id'] for info_dict in infos])
        # present_info_id = torch.Tensor(ids).to(device)
        
        # info_id_comp = old_info_id != present_info_id
        next_obs = next_obs.view(num_envs, obs_space_flat)
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)

    # for i, (po_r, a, r, id_) in enumerate(zip(processed_obs_r, actions, rewards, info_id_comps)):
    #     print(f"i : {i} po_r : {po_r} a : {a} r : {r}")
    #     if id_ == True:
    #         print(f"po_r : {po_r} a : {a} r : {r}")

        # if torch.numel(info_id_comp[info_id_comp == True]):
        #     print(f"old_info_id : {old_info_id}")
        #     print(f"present_info_id : {present_info_id}")
        #     print(f"info_id_comp : {info_id_comp}")
        #     print(f"rewards[step] : {rewards[step]}")

        for info in infos:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                break


    nextnonterminal = torch.zeros(num_envs).to(device)
    nextvalues = torch.zeros(num_envs).to(device)
    delta = torch.zeros(num_envs).to(device)
    

    with torch.no_grad():
        next_obs_ = preprocess_obs_r(next_obs)
        last_value = agent.get_value(next_obs_.to(device)).reshape(1, -1)
        last_value = last_value.squeeze(dim=0)
        if gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = torch.zeros(num_envs).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    # nextnonterminal[info_id_comps[t]] = 1.0 - next_done[info_id_comps[t]]
                    # nextvalues[info_id_comps[t]] = last_value[info_id_comps[t]]
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    # nextnonterminal[info_id_comps[t+1]] = 1.0 - dones[t+1][info_id_comps[t+1]]
                    # nextvalues[info_id_comps[t+1]] = values[t+1][info_id_comps[t+1]]
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                
                # delta[info_id_comps[t]] = rewards[t][info_id_comps[t]] + gamma * nextvalues[info_id_comps[t]] * nextnonterminal[info_id_comps[t]] - values[t][info_id_comps[t]]
                # advantages[t][info_id_comps[t]] = lastgaelam[info_id_comps[t]] = delta[info_id_comps[t]] + gamma * gae_lambda * nextnonterminal[info_id_comps[t]] * lastgaelam[info_id_comps[t]]
                delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_return = returns[t+1]
                returns[t] = rewards[t] + gamma * nextnonterminal * next_return
            advantages = returns - values

#     # print(f"advantages : {advantages}")
#     # print(f"returns : {returns}")

#     # print(f"obs[info_id_comps] : {obs[info_id_comps]}")
#     # print(f"info_id_comps[info_id_comps == True] : {info_id_comps[info_id_comps == True].shape}")
#     # print(f"obs[info_id_comps].shape : {obs[info_id_comps].shape}")

    b_obs = obs.reshape((num_steps*num_envs,obs_space_flat))
    b_preprocseed_obs = processed_obs.reshape((num_steps*num_envs, 8))
    b_preprocseed_obs_r = processed_obs_r.reshape((num_steps*num_envs, 4))
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,)+envs.action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    # b_info_id_comps = info_id_comps.reshape(-1)
    b_values = values.reshape(-1)

    # b_obs = obs[info_id_comps]
    # b_preprocseed_obs = processed_obs[info_id_comps]
    # b_preprocseed_obs_r = processed_obs_r[info_id_comps]
    # b_logprobs = logprobs[info_id_comps]
    # b_actions = actions[info_id_comps]
    # b_advantages = advantages[info_id_comps]
    # b_rewards = rewards[info_id_comps]
    # b_returns = returns[info_id_comps]
    # b_values = values[info_id_comps]

    # for i, (o, r, a) in enumerate(zip(b_preprocseed_obs_r, b_rewards, b_actions)):
    #     print(i, o, r, a)

#     # print(f"b_obs : {b_obs}")
#     # print(f"b_preprocseed_obs : {b_preprocseed_obs}")
#     # print(f"b_logprobs : {b_logprobs}")
#     # print(f"b_actions : {b_actions}")
#     # print(f"b_advantages : {b_advantages}")
#     # print(f"b_returns : {b_returns}")
#     # # print(f"b_info_id_comps : {b_info_id_comps.shape}")
#     # print(f"b_values : {b_values}")

    target_agent = AttAgent(envs).to(device)
    # batch_size = torch.numel(info_id_comps[info_id_comps == True])
    # minibatch_size = max(1, int(batch_size // num_minibatches))
    inds = np.arange(batch_size,)

    for i_epoch_pi in range(update_epochs):
        np.random.shuffle(inds)
        target_agent.load_state_dict(agent.state_dict())

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std(unbiased=False) + 1e-8)

            _, newlogproba, entropy = agent.get_action(b_preprocseed_obs_r[minibatch_ind], b_actions.long()[minibatch_ind])
            
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            new_values = agent.get_value(b_preprocseed_obs_r[minibatch_ind]).view(-1)
            if clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()

            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef 
            # print(f"loss : {loss}")

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        if kle_stop:
            if approx_kl > target_kl:
                break

        if kle_rollback:
            if (b_logprobs[minibatch_ind] - agent.get_action(b_preprocseed_obs_r[minibatch_ind], b_actions.long()[minibatch_ind])[1]).mean() > target_kl:
                agent.load_state_dict(target_agent.state_dict())
                break

    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if kle_stop or kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

torch.save(agent.state_dict(), f=save_path)
envs.close()
writer.close()
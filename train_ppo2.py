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
from utils.agent import CNNAgent, MLPAgent

exp_name = os.path.basename(__file__).rstrip(".py")
gym_id = "ebigfishl"
learning_rate = 2.5e-4
seed = 1
total_timesteps = int(1e9)
torch_deterministic = True
cuda = True
prod_mode = False
wandb_proj_name = "AttRL"
wandb_entity = None
capture_video = False
save_path = f"models/{exp_name}"

num_minibatches = 16
num_envs = 256*4
num_steps = 256 # the number of steps per game environment
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
envs = VecPyTorch(envs, device)
if capture_video:
    envs = VecVideoRecorder(envs, f'videos/{experiment_name}', 
                            record_video_trigger=lambda x: x % 300000 == 0, video_length=3000)

assert isinstance(envs.action_space, Discrete), "only discrete action space is supported"

agent = MLPAgent(envs=envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
if anneal_lr:
    lr = lambda f: f * learning_rate

obs_space_flat = np.array(envs.observation_space.shape).prod()

obs = torch.zeros((num_steps, num_envs, obs_space_flat)).to(device)
actions = torch.zeros((num_steps, num_envs) + envs.action_space.shape).to(device)
logprobs = torch.zeros((num_steps, num_envs)).to(device)
rewards = torch.zeros((num_steps, num_envs)).to(device)
dones = torch.zeros((num_steps, num_envs)).to(device)
values = torch.zeros((num_steps, num_envs)).to(device)

global_step = 0
start_time = time.time()

next_obs = envs.reset()
next_obs = next_obs.view(num_envs, np.array(envs.observation_space.shape).prod())
next_done = torch.zeros(num_envs).to(device)
num_updates = int(total_timesteps // batch_size)

for update in range(1, num_updates+1):
    if anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow
    for step in range(0, num_steps):
        global_step += 1 * num_envs
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            values[step] = agent.get_value(obs[step]).flatten()
            action, logproba, _ = agent.get_action(obs[step])

        actions[step] = action
        logprobs[step] = logproba

        next_obs, rs, ds, infos = envs.step(action)
        next_obs = next_obs.view(num_envs, obs_space_flat)
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds).to(device)

        for info in infos:
            if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                break
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)

        if gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
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

    b_obs = obs.reshape((num_steps*num_envs,obs_space_flat))
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,)+envs.action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    target_agent = MLPAgent(envs).to(device)
    inds = np.arange(batch_size,)

    for i_epoch_pi in range(update_epochs):
        np.random.shuffle(inds)
        target_agent.load_state_dict(agent.state_dict())
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _, newlogproba, entropy = agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-clip_coef, 1+clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
            if clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -clip_coef, clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()

            loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

    if kle_stop:
            if approx_kl > target_kl:
                break
    if kle_rollback:
        if (b_logprobs[minibatch_ind] - agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])[1]).mean() > target_kl:
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
#IMPORTS

#HYPERPARAMETERS
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from distutils.util import strtobool
import numpy as np
from procgen import ProcgenEnv
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder, VecFrameStack, VecExtractDictObs, VecMonitor
from utils.wrappers import VecPyTorch
from utils.agent import CNNAgent, MLPAgent, DDTAgent, DDTAgentNew

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="ebigfishs",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=int(1e5),
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="AttRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    #DDT specific argument
    parser.add_argument('--num_leaves', type=int, default=8, help="number of leaves for DDT/DRL ")

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=8,
                        help='the number of mini batch')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='the number of parallel game environment')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.999,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=3,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                         help='Use GAE for advantage computation')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())

args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)

experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
device = torch.device('cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

env = ProcgenEnv(num_envs=args.num_envs, env_name=args.gym_id, num_levels=0, start_level=0, distribution_mode='hard')
env = VecExtractDictObs(env, "positions")
env = VecFrameStack(env, n_stack=2)
env = VecMonitor(venv=env)
env = VecNormalize(venv=env, norm_obs=False)
env = VecPyTorch(env, device)
if args.capture_video:
    envs = VecVideoRecorder(env, f'videos/{experiment_name}', 
                            record_video_trigger=lambda x: x % 1000000== 0, video_length=100)
assert isinstance(env.action_space, Discrete), "only discrete action space is supported"


### CHANGE TO DDTAGENT
agent = DDTAgentNew(bot_name="ddtebigfishs", input_dim=env.observation_space.shape[0] * env.observation_space.shape[1], output_dim=env.action_space.n, rule_list=False, num_rules=8)
# optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    lr = lambda f: f * args.learning_rate

obs_space_flat = np.array(env.observation_space.shape).prod()

obs = torch.zeros((args.num_steps, args.num_envs, obs_space_flat)).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + env.action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# states_list = torch.zeros((args.num_steps, args.num_envs, obs_space_flat))
# action_probs_list = torch.zeros((args.num_steps, args.num_envs, env.action_space.n))
# value_list = torch.zeros((args.num_steps, args.num_envs))
# hidden_state_list = torch.zeros((args.num_steps, args.num_envs))
# rewards_list = torch.zeros((args.num_steps, args.num_envs))
# deeper_value_list = torch.zeros((args.num_steps, args.num_envs))
# deeper_action_list = torch.zeros((args.num_steps, args.num_envs))
# deeper_advantage_list = torch.zeros((args.num_steps, args.num_envs))
# action_taken_list = torch.zeros((args.num_steps, args.num_envs))
# advantage_list = torch.zeros((args.num_steps, args.num_envs))
# full_probs_list = torch.zeros((args.num_steps, args.num_envs, env.action_space.n))
# deeper_full_probs_list = torch.zeros((args.num_steps, args.num_envs))

global_step = 0
start_time = time.time()

global_step = 0
start_time = time.time()

next_obs = env.reset()
next_obs = next_obs.view(args.num_envs, env.observation_space.shape[0]*env.observation_space.shape[1]) # change after using pytorch wrapper
# next_obs = next_obs.reshape(args.num_envs, env.observation_space.shape[0]*env.observation_space.shape[1])
next_done = torch.zeros(args.num_envs)
num_updates = int(args.total_timesteps // args.batch_size)

print(num_updates)
for update in range(1, num_updates+1):
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        agent.ppo.actor_opt.param_groups[0]['lr'] = lrnow
        agent.ppo.critic_opt.param_groups[0]['lr'] = lrnow

    # COLLECT ROLLOUTS
    for step in range(0, args.num_steps):
        global_step += args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        with torch.no_grad():
            # values[step] = agent.get_value(obs[step]).flatten()
            # action, logproba, _ = agent.get_action(obs[step])
            action= agent.get_action(obs[step])

        actions[step] = action
        # logprobs[step] = logproba

        next_obs, rs, ds, infos = env.step(action)
        next_obs = next_obs.view(args.num_envs, obs_space_flat)
        rewards[step], next_done = rs.view(-1), torch.Tensor(ds)

    # ADVANTAGE CALCULATION
    with torch.no_grad():
        last_value = agent.get_value(next_obs.to(device)).reshape(1, -1)
        if args.gae:
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    nextvalues = values[t+1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = last_value
                else:
                    nextnonterminal = 1.0 - dones[t+1]
                    next_return = returns[t+1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
            advantages = returns - values
    print(update)
    # PPO MINIBATCH UPDATE
    b_obs = obs.reshape((args.num_steps*args.num_envs,obs_space_flat))
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,)+env.action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    target_agent = DDTAgentNew(bot_name="ddtebigfishs", input_dim=env.observation_space.shape[0] * env.observation_space.shape[1], output_dim=env.action_space.n, rule_list=False, num_rules=8)
    inds = np.arange(args.batch_size,)

    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)
        # target_agent.load_state_dict(agent.state_dict())
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]
            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            _, newlogproba, entropy = agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            new_values = agent.get_value(b_obs[minibatch_ind]).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef, args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind])**2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2).mean()

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

        nn.utils.clip_grad_norm_(agent.ppo.critic.parameters(), args.max_grad_norm)
        nn.utils.clip_grad_norm_(agent.ppo.actor.parameters(), args.max_grad_norm)
        agent.ppo.critic_opt.zero_grad()
        agent.ppo.actor_opt.zero_grad()
        loss.backward()
        agent.ppo.critic_opt.step()
        agent.ppo.actor_opt.step()

        # optimizer.zero_grad()
        # loss.backward()
        # nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        # optimizer.step()

    # KLE KRAPPA
    if args.kle_stop:
            if approx_kl > args.target_kl:
                break
    if args.kle_rollback:
        if (b_logprobs[minibatch_ind] - agent.get_action(b_obs[minibatch_ind], b_actions.long()[minibatch_ind])[1]).mean() > args.target_kl:
            agent.load_state_dict(target_agent.state_dict())
            break


    # LOGGING KRAPPA
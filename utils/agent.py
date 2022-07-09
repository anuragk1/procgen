import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class MLPAgent(nn.Module):
    def __init__(self, envs):
        super(MLPAgent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), 32)),
            nn.Tanh(),
            # layer_init(nn.Linear(32, 32)),
            # nn.ReLU(),
            # layer_init(nn.Linear(64, 128)),
            # nn.ReLU(),
            # layer_init(nn.Linear(128, 128)),
            # nn.ReLU(),
            # layer_init(nn.Linear(128, 64)),
            # nn.ReLU(),
            layer_init(nn.Linear(32, 32)),
            nn.Tanh()
        )
        self.actor = layer_init(nn.Linear(32, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(32, 1), std=1)

    def forward(self, x):
        return self.network(x) 

    def get_action(self, x, action=None):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(self.forward(x))

class CNNAgent(nn.Module):
    def __init__(self, envs, channels=3):
        super(CNNAgent, self).__init__()
        self.network = nn.Sequential(
            Scale(1/255),
            layer_init(nn.Conv2d(channels, 32, 4, stride=3)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(8*8*32, 512)),
            nn.ReLU()
        )
        self.actor = layer_init(nn.Linear(512, envs.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x):
        return self.network(x.permute((0, 3, 1, 2))) # "bhwc" -> "bchw"

    def get_action(self, x, action=None):
        logits = self.actor(self.forward(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()

    def get_value(self, x):
        return self.critic(self.forward(x))
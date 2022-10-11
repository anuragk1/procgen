from xxlimited import new
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# from utils.misc import onehot_from_logits, categorical_sample

cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')

def preprocess_obs(obs):
    new_obs = obs
    new_obs = new_obs.cpu().detach().numpy()
    del_idx = np.arange(2, new_obs[0].size, 3)
    new_obs = np.delete(new_obs, del_idx, axis=1)
    new_obs = torch.Tensor(new_obs).to(device)
    
    return new_obs

def preprocess_obs_r(obs):
    new_obs = obs
    new_obs = new_obs.cpu().detach().numpy()
    idxs = np.arange(new_obs[0].size)
    del_idx = np.arange(2, new_obs[0].size, 3)
    idxs = np.delete(idxs, del_idx)
    new_obs = np.delete(new_obs, idxs, axis=1)
    new_obs = torch.Tensor(new_obs).to(device)
    
    return new_obs

def split_obs(obs):
    obs_split = obs
    obs_split = obs_split.cpu().detach().numpy()
    obs_split = [np.hsplit(x, 2) for x in np.vsplit(obs_split, 1)]
    obs_split = torch.Tensor(obs_split).squeeze().to(device)

    return obs_split

# Change the functions to make them compatible with PPO
def onehot_from_logits(logits, eps=0.0, dim=1):
    argmax_acs = (logits == logits.max(dim, keepdim=True)[0]).float()

    if eps == 0.0:
        return argmax_acs
    rand_acs = Variable(torch.eye(logits.shape[1])[[np.random.choice(
        range(logits.shape[1]), size=logits.shape[0])]], requires_grad=False)
    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])

def categorical_sample(probs, use_cuda=False):
    int_acs = torch.multinomial(probs, 1)
    if use_cuda:
        tensor_type = torch.cuda.FloatTensor
    else:
        tensor_type = torch.FloatTensor
    acs = Variable(tensor_type(*probs.shape).fill_(0)).scatter_(1, int_acs, 1)
    return int_acs, acs
###################

class BasePolicy(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=32, nonlin=F.leaky_relu, norm_in=False, onehot_dim=0):
        super(BasePolicy, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin

    def forward(self, x):
        onehot = None
        if type(X) is tuple:
            X, onehot = X
        inp = self.in_fn(X)  # don't batchnorm onehot
        if onehot is not None:
            inp = torch.cat((onehot, inp), dim=2)
        h1 = self.nonlin(self.fc1(inp))
        h2 = self.nonlin(self.fc2(h1))
        out = self.fc3(h2)
        return out

class DiscretePolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super(DiscretePolicy, self).__init__(*args, **kwargs)

    def forward(self, obs, sample=True, return_all_probs=False, return_log_pi=False, regularize=False, return_entropy=False):
        out = super(DiscretePolicy, self).forward(obs)
        probs = F.softmax(out, dim=1)
        on_gpu = next(self.parameters()).is_cuda
        if sample:
            int_act, act = categorical_sample(probs, use_cuda=on_gpu)
        else:
            act = onehot_from_logits(probs)

        rets = [act]
        if return_log_pi or return_entropy:
            log_probs = F.log_softmax(out, dim=1)
        if return_all_probs:
            rets.append(probs)
        if return_log_pi:
            # return log probability of selected action
            rets.append(log_probs.gather(1, int_act))
        if regularize:
            rets.append([(out**2).mean()])
        if return_entropy:
            rets.append(-(log_probs * probs).sum(1).mean())
        if len(rets) == 1:
            return rets[0]
        return rets

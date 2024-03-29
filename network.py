# Networks for value / policy / discriminator
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from gym.spaces import Box, Discrete

class MLP(nn.Module):
    def __init__(self, layers, activation=torch.tanh, output_activation=None,
                 output_squeeze=False):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.output_activation = output_activation
        self.output_squeeze = output_squeeze
        
        for i, layer in enumerate(layers[1:]):
            self.layers.append(nn.Linear(layers[i], layer))
            nn.init.zeros_(self.layers[i].bias)

    def forward(self, input):
        x = input
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation is None:
            x = self.layers[-1](x)
        else:
            x = self.output_activation(self.layers[-1](x))
        return x.squeeze() if self.output_squeeze else x

class BiclassificationPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation):
        super(BiclassificationPolicy, self).__init__()

        self.output_dim = 2
        self.logits = MLP(layers=[input_dim] + list(hidden_dims) + [self.output_dim], activation=activation)

    def forward(self, x, label=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        l = policy.sample()
        logp_l = policy.log_prob(l).squeeze()
        if label is not None:
            logp = policy.log_prob(label).squeeze()
        else:
            logp = None

        return l, logp, logp_l

class GaussianPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, action_dim):
        super(GaussianPolicy, self).__init__()

        print("Guassian policy used.")
        self.mu = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation, output_activation=output_activation)
        self.log_std = nn.Parameter(-0.5 * torch.ones(action_dim))

    def forward(self, x, a=None):
        policy = Normal(self.mu(x), self.log_std.exp())
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None

        return pi, logp, logp_pi

class CategoricalPolicy(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation, output_activation, action_dim):
        super(CategoricalPolicy, self).__init__()

        print("Categorical policy used.")
        self.logits = MLP(layers=[input_dim] + list(hidden_dims) + [action_dim], activation=activation)

    def forward(self, x, a=None):
        logits = self.logits(x)
        policy = Categorical(logits=logits)
        pi = policy.sample()
        logp_pi = policy.log_prob(pi).squeeze()
        if a is not None:
            logp = policy.log_prob(a).squeeze()
        else:
            logp = None

        return pi, logp, logp_pi

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_space, hidden_dims=(64, 64), activation=torch.tanh, output_activation=None, policy=None):
        super(ActorCritic, self).__init__()

        if policy is None:
            if isinstance(action_space, Box):
                self.policy = GaussianPolicy(input_dim, hidden_dims, activation, output_activation, action_space.shape[0])
            elif isinstance(action_space, Discrete):
                self.policy = CategoricalPolicy(input_dim, hidden_dims, activation, output_activation, action_space.n)
        else: 
            self.policy = policy(input_dim, hidden_dims, activation, output_activation, action_space)

        self.value_f = MLP(layers=[input_dim] + list(hidden_dims) + [1], activation=activation, output_squeeze=True)

    def forward(self, x, a=None):
        pi, logp, logp_pi = self.policy(x, a)
        v = self.value_f(x)

        return pi, logp, logp_pi, v

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 64), activation=torch.relu, output_activation=torch.softmax):
        super(Discriminator, self).__init__()

        self.policy = BiclassificationPolicy(input_dim, hidden_dims, activation, output_activation)

    def forward(self, state, gt=None):
        label, loggt, logp = self.policy(state, gt)
        return label, loggt, logp

def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
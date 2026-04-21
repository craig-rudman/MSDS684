import torch
import torch.nn as nn

def _mlp_body(input_dim, hidden_sizes, activation=nn.ReLU):
    layers = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), activation()]
        in_dim = h
    return nn.Sequential(*layers)


class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes=(64, 64)):
        super().__init__()
        self.trunk = _mlp_body(obs_dim, hidden_sizes)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs):
        return self.value_head(self.trunk(obs)).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 64), state_dependent_std=False):
        super().__init__()
        self.state_dependent_std = state_dependent_std
        self.trunk = _mlp_body(obs_dim, hidden_sizes)
        self.mu_head = nn.Linear(hidden_sizes[-1], act_dim)
        if state_dependent_std:
            self.log_std_head = nn.Linear(hidden_sizes[-1], act_dim)
        else:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        features = self.trunk(obs)
        mu = self.mu_head(features)
        if self.state_dependent_std:
            log_std = self.log_std_head(features).clamp(-20, 2)
        else:
            log_std = self.log_std.expand_as(mu).clamp(-20, 2)
        return mu, log_std

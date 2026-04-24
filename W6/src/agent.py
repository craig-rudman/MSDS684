import torch
import torch.nn as nn
from torch.distributions import Normal

def _mlp_body(input_dim, hidden_sizes, activation=nn.ReLU):
    layers = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), activation()]
        in_dim = h
    return nn.Sequential(*layers)


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(128, 128)):
        super().__init__()
        self.trunk = _mlp_body(obs_dim, hidden_sizes)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs):
        return self.value_head(self.trunk(obs)).squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128, 128)):
        super().__init__()
        self.trunk = _mlp_body(obs_dim, hidden_sizes)
        self.mu_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        mu = self.mu_head(self.trunk(obs))
        log_std = self.log_std.expand_as(mu).clamp(-20, 2)
        return mu, log_std


class Agent:
    def __init__(self, obs_dim, act_dim, actor_lr, critic_lr, gamma, act_low, act_high,
                 hidden_sizes=(64, 64)):
        self.gamma = gamma
        self.act_low = act_low
        self.act_high = act_high

        self.actor = Actor(obs_dim, act_dim, hidden_sizes=hidden_sizes)
        self.critic = Critic(obs_dim, hidden_sizes=hidden_sizes)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self._config = dict(
            obs_dim=obs_dim, act_dim=act_dim,
            actor_lr=actor_lr, critic_lr=critic_lr,
            gamma=gamma, act_low=act_low, act_high=act_high,
            hidden_sizes=hidden_sizes,
        )

    def get_value(self, obs):
        return self.critic(obs)

    def select_action(self, obs):
        mu, log_std = self.actor(obs)
        dist = Normal(mu, log_std.exp())
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        action = torch.clamp(action, self.act_low, self.act_high)
        value = self.critic(obs)
        return action, log_prob, value, entropy

    def update(self, log_prob, value, next_value, reward, done):
        target = reward + self.gamma * next_value.detach() * (1 - float(done))
        delta = target - value

        critic_loss = delta.pow(2)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -(log_prob * delta.detach())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
                "td_error": delta.item()}

    def save(self, path):
        torch.save({"config": self._config,
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, weights_only=False)
        agent = cls(**checkpoint["config"])
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        return agent

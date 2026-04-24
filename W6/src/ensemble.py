import torch


def ensemble_mu(actors, obs):
    with torch.no_grad():
        mus = [actor(obs)[0] for actor in actors]
    return torch.stack(mus).mean(dim=0)

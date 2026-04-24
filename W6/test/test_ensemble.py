import pytest
import torch

OBS_DIM = 8
ACT_DIM = 2


@pytest.fixture
def two_actors():
    from src.agent import Actor
    torch.manual_seed(0)
    a1 = Actor(OBS_DIM, ACT_DIM)
    torch.manual_seed(1)
    a2 = Actor(OBS_DIM, ACT_DIM)
    return [a1, a2]


def test_ensemble_mu_is_mean_of_actor_mus(two_actors):
    from src.ensemble import ensemble_mu
    obs = torch.zeros(OBS_DIM)
    with torch.no_grad():
        mu1, _ = two_actors[0](obs)
        mu2, _ = two_actors[1](obs)
        expected = (mu1 + mu2) / 2
        result = ensemble_mu(two_actors, obs)
    assert result.shape == expected.shape
    assert torch.allclose(result, expected)


def test_ensemble_mu_single_actor_returns_its_mu():
    from src.agent import Actor
    from src.ensemble import ensemble_mu
    torch.manual_seed(0)
    actor = Actor(OBS_DIM, ACT_DIM)
    obs = torch.ones(OBS_DIM)
    with torch.no_grad():
        expected, _ = actor(obs)
        result = ensemble_mu([actor], obs)
    assert torch.allclose(result, expected)


def test_ensemble_mu_batched_obs(two_actors):
    from src.ensemble import ensemble_mu
    obs = torch.zeros(4, OBS_DIM)
    result = ensemble_mu(two_actors, obs)
    assert result.shape == (4, ACT_DIM)

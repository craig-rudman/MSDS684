import pytest
import torch

OBS_DIM = 8
ACT_DIM = 2


@pytest.fixture
def obs():
    return torch.zeros(OBS_DIM)


@pytest.fixture
def batch_obs():
    return torch.zeros(4, OBS_DIM)


@pytest.fixture
def actor():
    from src.agent import Actor
    return Actor(OBS_DIM, ACT_DIM)


class TestActorOutputType:
    def test_forward_returns_tuple(self, actor, obs):
        out = actor(obs)
        assert isinstance(out, tuple) and len(out) == 2

    def test_forward_returns_tensors(self, actor, obs):
        mu, log_std = actor(obs)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(log_std, torch.Tensor)


class TestActorOutputShape:
    def test_mu_shape_single_obs(self, actor, obs):
        mu, _ = actor(obs)
        assert mu.shape == (ACT_DIM,)

    def test_log_std_shape_single_obs(self, actor, obs):
        _, log_std = actor(obs)
        assert log_std.shape == (ACT_DIM,)

    def test_mu_shape_batch(self, actor, batch_obs):
        mu, _ = actor(batch_obs)
        assert mu.shape == (4, ACT_DIM)

    def test_log_std_shape_batch(self, actor, batch_obs):
        _, log_std = actor(batch_obs)
        assert log_std.shape == (4, ACT_DIM)


class TestActorLogStd:
    def test_sigma_positive(self, actor, obs):
        _, log_std = actor(obs)
        assert (log_std.exp() > 0).all()

    def test_log_std_finite(self, actor, obs):
        _, log_std = actor(obs)
        assert torch.isfinite(log_std).all()


class TestActorLogStdDesign:
    def test_log_std_state_independent(self, actor):
        obs1 = torch.zeros(OBS_DIM)
        obs2 = torch.ones(OBS_DIM)
        _, log_std1 = actor(obs1)
        _, log_std2 = actor(obs2)
        assert torch.allclose(log_std1, log_std2)

    def test_log_std_is_nn_parameter(self, actor):
        param_names = [name for name, _ in actor.named_parameters()]
        assert any("log_std" in name for name in param_names)

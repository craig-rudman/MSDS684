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
def actor_head():
    from src.agent import Actor
    return Actor(OBS_DIM, ACT_DIM, state_dependent_std=True)


@pytest.fixture
def actor_free():
    from src.agent import Actor
    return Actor(OBS_DIM, ACT_DIM, state_dependent_std=False)


class TestActorOutputType:
    def test_forward_returns_tuple(self, actor_head, obs):
        out = actor_head(obs)
        assert isinstance(out, tuple) and len(out) == 2

    def test_forward_returns_tensors(self, actor_head, obs):
        mu, log_std = actor_head(obs)
        assert isinstance(mu, torch.Tensor)
        assert isinstance(log_std, torch.Tensor)


class TestActorOutputShape:
    def test_mu_shape_single_obs(self, actor_head, obs):
        mu, _ = actor_head(obs)
        assert mu.shape == (ACT_DIM,)

    def test_log_std_shape_single_obs(self, actor_head, obs):
        _, log_std = actor_head(obs)
        assert log_std.shape == (ACT_DIM,)

    def test_mu_shape_batch(self, actor_head, batch_obs):
        mu, _ = actor_head(batch_obs)
        assert mu.shape == (4, ACT_DIM)

    def test_log_std_shape_batch(self, actor_head, batch_obs):
        _, log_std = actor_head(batch_obs)
        assert log_std.shape == (4, ACT_DIM)


class TestActorLogStd:
    def test_sigma_positive_head(self, actor_head, obs):
        _, log_std = actor_head(obs)
        assert (log_std.exp() > 0).all()

    def test_sigma_positive_free(self, actor_free, obs):
        _, log_std = actor_free(obs)
        assert (log_std.exp() > 0).all()

    def test_log_std_finite(self, actor_head, obs):
        _, log_std = actor_head(obs)
        assert torch.isfinite(log_std).all()


class TestActorSigmaMode:
    def test_free_param_log_std_state_independent(self, actor_free):
        obs1 = torch.zeros(OBS_DIM)
        obs2 = torch.ones(OBS_DIM)
        _, log_std1 = actor_free(obs1)
        _, log_std2 = actor_free(obs2)
        assert torch.allclose(log_std1, log_std2)

    def test_head_log_std_state_dependent(self, actor_head):
        torch.manual_seed(0)
        obs1 = torch.randn(OBS_DIM)
        obs2 = torch.randn(OBS_DIM)
        _, log_std1 = actor_head(obs1)
        _, log_std2 = actor_head(obs2)
        assert not torch.allclose(log_std1, log_std2)

    def test_free_param_log_std_is_nn_parameter(self, actor_free):
        param_names = [name for name, _ in actor_free.named_parameters()]
        assert any("log_std" in name for name in param_names)

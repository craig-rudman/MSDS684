import pytest
import torch

OBS_DIM = 8


@pytest.fixture
def critic():
    from src.agent import Critic
    return Critic(OBS_DIM)


@pytest.fixture
def obs():
    return torch.zeros(OBS_DIM)


@pytest.fixture
def batch_obs():
    return torch.zeros(4, OBS_DIM)


class TestCriticOutputType:
    def test_forward_returns_tensor(self, critic):
        v = critic(torch.zeros(OBS_DIM))
        assert isinstance(v, torch.Tensor)


class TestCriticGradient:
    def test_output_has_grad_fn(self, critic, obs):
        v = critic(obs)
        assert v.grad_fn is not None


class TestCriticOutputShape:
    def test_scalar_single_obs(self, critic, obs):
        v = critic(obs)
        assert v.shape == ()

    def test_batch_shape(self, critic, batch_obs):
        v = critic(batch_obs)
        assert v.shape == (4,)


class TestCriticOutputValue:
    def test_output_finite(self, critic, obs):
        v = critic(obs)
        assert torch.isfinite(v)

    def test_batch_output_finite(self, critic, batch_obs):
        v = critic(batch_obs)
        assert torch.isfinite(v).all()

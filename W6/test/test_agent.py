import pytest
import torch
import os
import tempfile

OBS_DIM = 8
ACT_DIM = 2
ACT_LOW = torch.tensor([-1.0, -1.0])
ACT_HIGH = torch.tensor([1.0, 1.0])


@pytest.fixture
def agent():
    from src.agent import Agent
    return Agent(
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        actor_lr=3e-4,
        critic_lr=1e-3,
        gamma=0.99,
        act_low=ACT_LOW,
        act_high=ACT_HIGH,
    )


@pytest.fixture
def obs():
    return torch.zeros(OBS_DIM)


class TestAgentSelectAction:
    def test_returns_three_values(self, agent, obs):
        result = agent.select_action(obs)
        assert len(result) == 3

    def test_action_shape(self, agent, obs):
        action, _, _ = agent.select_action(obs)
        assert action.shape == (ACT_DIM,)

    def test_action_within_bounds(self, agent, obs):
        action, _, _ = agent.select_action(obs)
        assert (action >= ACT_LOW).all()
        assert (action <= ACT_HIGH).all()

    def test_log_prob_scalar(self, agent, obs):
        _, log_prob, _ = agent.select_action(obs)
        assert log_prob.shape == ()

    def test_log_prob_finite(self, agent, obs):
        _, log_prob, _ = agent.select_action(obs)
        assert torch.isfinite(log_prob)

    def test_value_scalar(self, agent, obs):
        _, _, value = agent.select_action(obs)
        assert value.shape == ()

    def test_value_finite(self, agent, obs):
        _, _, value = agent.select_action(obs)
        assert torch.isfinite(value)


class TestAgentUpdate:
    def test_update_returns_losses(self, agent, obs):
        action, log_prob, value = agent.select_action(obs)
        next_obs = torch.randn(OBS_DIM)
        _, _, next_value = agent.select_action(next_obs)
        result = agent.update(log_prob, value, next_value, reward=1.0, done=False)
        assert "actor_loss" in result and "critic_loss" in result

    def test_actor_params_change_after_update(self, agent, obs):
        params_before = [p.clone() for p in agent.actor.parameters()]
        action, log_prob, value = agent.select_action(obs)
        _, _, next_value = agent.select_action(torch.randn(OBS_DIM))
        agent.update(log_prob, value, next_value, reward=1.0, done=False)
        params_after = list(agent.actor.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))

    def test_critic_params_change_after_update(self, agent, obs):
        params_before = [p.clone() for p in agent.critic.parameters()]
        action, log_prob, value = agent.select_action(obs)
        _, _, next_value = agent.select_action(torch.randn(OBS_DIM))
        agent.update(log_prob, value, next_value, reward=1.0, done=False)
        params_after = list(agent.critic.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(params_before, params_after))

    def test_done_zeroes_next_value(self, agent, obs):
        _, log_prob, value = agent.select_action(obs)
        value_scalar = value.item()
        large_next_value = torch.tensor(1000.0)
        result = agent.update(log_prob, value, large_next_value, reward=1.0, done=True)
        expected = (1.0 - value_scalar) ** 2
        assert abs(result["critic_loss"] - expected) < 1e-4


class TestAgentSaveLoad:
    def test_save_creates_file(self, agent, tmp_path):
        path = str(tmp_path / "agent.pt")
        agent.save(path)
        assert os.path.exists(path)

    def test_load_restores_action(self, agent, obs):
        from src.agent import Agent
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "agent.pt")
            agent.save(path)
            loaded = Agent.load(path)
            torch.manual_seed(0)
            action1, _, _ = agent.select_action(obs)
            torch.manual_seed(0)
            action2, _, _ = loaded.select_action(obs)
            assert torch.allclose(action1, action2)

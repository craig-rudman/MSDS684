import math


class LinearDecay:
    """Decay epsilon by a fixed amount each step."""

    def __init__(self, eps_min: float = 0.01, decay_rate: float = 0.0001):
        self.eps_min = eps_min
        self.decay_rate = decay_rate

    def __call__(self, eps: float) -> float:
        return max(self.eps_min, eps - self.decay_rate)


class ExponentialDecay:
    """Decay epsilon by a multiplicative factor each step."""

    def __init__(self, eps_min: float = 0.01, decay_factor: float = 0.99999):
        self.eps_min = eps_min
        self.decay_factor = decay_factor

    def __call__(self, eps: float) -> float:
        return max(self.eps_min, eps * self.decay_factor)


class CosineDecay:
    """Cosine annealing from eps_start to eps_min over total_steps."""

    def __init__(self, eps_start: float = 1.0, eps_min: float = 0.01,
                 total_steps: int = 500_000):
        self.eps_start = eps_start
        self.eps_min = eps_min
        self.total_steps = total_steps
        self.step = 0

    def __call__(self, eps: float) -> float:
        self.step += 1
        progress = min(self.step / self.total_steps, 1.0)
        return self.eps_min + 0.5 * (self.eps_start - self.eps_min) * (
            1 + math.cos(math.pi * progress))


class StepDecay:
    """Drop epsilon by a factor at fixed intervals."""

    def __init__(self, eps_min: float = 0.01, drop_factor: float = 0.5,
                 drop_every: int = 100_000):
        self.eps_min = eps_min
        self.drop_factor = drop_factor
        self.drop_every = drop_every
        self.step = 0
        self.current_eps = None

    def __call__(self, eps: float) -> float:
        if self.current_eps is None:
            self.current_eps = eps
        self.step += 1
        if self.step % self.drop_every == 0:
            self.current_eps = max(self.eps_min, self.current_eps * self.drop_factor)
        return self.current_eps


class AdaptiveDecay:
    """Decay epsilon based on the agent's exploration signal.

    High signal (many visits) -> decay faster.
    Low signal (few visits) -> hold epsilon steady.
    """

    def __init__(self, eps_min: float = 0.01, agent=None,
                 sensitivity: float = 0.01):
        self.eps_min = eps_min
        self.agent = agent
        self.sensitivity = sensitivity

    def __call__(self, eps: float) -> float:
        signal = self.agent.get_exploration_signal()
        decay_amount = self.sensitivity * signal
        return max(self.eps_min, eps - decay_amount)

import gymnasium as gym
import numpy as np
import pytest
import torch

from src.dqn import (SETTINGS, DQNAgent, ReplayBuffer, epsilon, linear, load_network, make_network, network_policy,
                     td_targets, train)

GAMMA = 0.9
# small settings so that the tiny MDP trains in a few seconds
TINY = {**SETTINGS, "hidden": 16, "lr_start": 1e-3, "lr_end": 1e-3, "batch_size": 32, "buffer_size": 5_000,
        "learn_start": 500, "target_every": 200, "eps_decay_steps": 2_000, "eps_end": 0.1}


class TinyEnv:
    # the two state MDP of test_tabular.py with the state given as a one hot vector. In state 0: action 0
    # gives 1 and stays, action 1 gives 0 and moves to state 1 but takes 3 weeks. In state 1: action 0 gives 2
    # and stays, action 1 gives 0 and goes back. Every reward gets noise with mean 0. Episodes end by time limit
    observation_space = gym.spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None):
        self.s, self.t = 0, 0
        self.rng = np.random.default_rng(seed)
        return self._obs(), {}

    def step(self, a):
        reward = [[1.0, 0.0], [2.0, 0.0]][self.s][a] + self.rng.normal(0, 1.0)
        weeks = 3 if (self.s, a) == (0, 1) else 1
        self.s = self.s if a == 0 else 1 - self.s
        self.t += 1
        return self._obs(), reward, False, self.t >= 50, {"weeks": weeks}

    def _obs(self):
        return np.eye(2, dtype=np.float32)[self.s]


def tiny_q_star(gamma):
    # by hand: in state 1 staying is best, so V1 = 2 / (1 - gamma). In state 0 moving is best when gamma^3 V1 > 1 / (1 - gamma)
    v1 = 2 / (1 - gamma)
    v0 = max(1 / (1 - gamma), gamma**3 * v1)
    return np.array([[1 + gamma * v0, gamma**3 * v1], [2 + gamma * v1, gamma * v0]])


class FixedQ(torch.nn.Module):
    # a fake network that always returns the same Q values, one row per next state
    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values)

    def forward(self, x):
        return self.values


def test_network_shapes():
    net = make_network(6, 3, 64)
    assert net(torch.zeros(5, 6)).shape == (5, 3)
    assert net(torch.zeros(6)).shape == (3,)
    assert sum(p.numel() for p in net.parameters()) == 6 * 64 + 64 + 64 * 64 + 64 + 64 * 3 + 3


def test_saved_network_loads_back(tmp_path):
    net = make_network(6, 3, 32)
    torch.save({"state": net.state_dict(), "settings": {**SETTINGS, "hidden": 32}}, tmp_path / "net.pt")
    loaded = load_network(tmp_path / "net.pt")
    x = torch.randn(10, 6)
    assert torch.equal(net(x), loaded(x))


def test_buffer_keeps_the_last_steps():
    buffer = ReplayBuffer(3, 2)
    for i in range(5):
        buffer.add([i, i], i % 3, float(i), 1 + i, [i + 1, i + 1], 0.0)
    assert len(buffer) == 3
    assert sorted(buffer.rewards) == [2.0, 3.0, 4.0]
    assert sorted(buffer.weeks) == [3.0, 4.0, 5.0]
    assert np.array_equal(buffer.next_obs[buffer.rewards.argmax()], [5, 5])


def test_buffer_sample_types():
    buffer = ReplayBuffer(10, 6)
    for i in range(4):
        buffer.add(np.ones(6) * i, 1, -0.5, 2, np.ones(6), 0.0)
    obs, actions, rewards, weeks, next_obs, terminated = buffer.sample(32)
    assert obs.shape == (32, 6) and next_obs.shape == (32, 6)
    assert actions.dtype == torch.int64 and obs.dtype == torch.float32
    assert set(obs[:, 0].tolist()) <= {0.0, 1.0, 2.0, 3.0}
    assert torch.all(weeks == 2) and torch.all(rewards == -0.5)


def test_td_targets_by_hand():
    target = FixedQ([[1.0, 5.0, 2.0], [3.0, 0.0, 4.0], [1.0, 1.0, 1.0]])
    online = FixedQ([[0.0, 0.0, 9.0], [9.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rewards = torch.tensor([-1.0, -2.0, -3.0])
    weeks = torch.tensor([1.0, 3.0, 1.0])
    terminated = torch.tensor([0.0, 0.0, 1.0])
    y = td_targets(online, target, rewards, weeks, None, terminated, 0.9, double=False)
    assert torch.allclose(y, torch.tensor([-1 + 0.9 * 5, -2 + 0.9**3 * 4, -3.0]))
    # double DQN: online picks action 2 in row 0 and action 0 in row 1, the target network gives their values
    y = td_targets(online, target, rewards, weeks, None, terminated, 0.9, double=True)
    assert torch.allclose(y, torch.tensor([-1 + 0.9 * 2, -2 + 0.9**3 * 3, -3.0]))


def test_epsilon_schedule():
    s = {"eps_start": 1.0, "eps_end": 0.02, "eps_decay_steps": 100}
    assert epsilon(0, s) == 1.0
    assert np.isclose(epsilon(50, s), 0.51)
    assert np.isclose(epsilon(100, s), 0.02) and np.isclose(epsilon(10_000, s), 0.02)
    assert np.isclose(linear(5e-4, 5e-5, 200_000, 400_000), 2.75e-4)


def test_learning_rate_falls_during_training():
    agent = DQNAgent(2, 2, GAMMA, {**TINY, "learn_start": 1, "lr_start": 1e-3, "lr_end": 1e-4, "lr_decay_steps": 100})
    env = TinyEnv()
    obs, _ = env.reset(seed=0)
    rates = {}
    for step in range(1, 201):
        action = agent.act(obs, eps=1.0)
        next_obs, reward, terminated, _, info = env.step(action)
        agent.observe(obs, action, reward, info["weeks"], next_obs, terminated)
        obs = next_obs
        rates[step] = agent.optimizer.param_groups[0]["lr"]
    assert np.isclose(rates[1], 1e-3 - 0.01 * 9e-4) and np.isclose(rates[50], 5.5e-4)
    assert np.isclose(rates[100], 1e-4) and np.isclose(rates[200], 1e-4)


def test_target_network_is_copied_on_schedule():
    agent = DQNAgent(2, 2, GAMMA, {**TINY, "learn_start": 10, "target_every": 50})
    env = TinyEnv()
    obs, _ = env.reset(seed=0)
    same = []
    for step in range(1, 101):
        action = agent.act(obs, eps=1.0)
        next_obs, reward, terminated, _, info = env.step(action)
        agent.observe(obs, action, reward, info["weeks"], next_obs, terminated)
        obs = next_obs
        same.append(all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters())))
    # the networks are equal until training starts at step 10, then only right after each copy at steps 50 and 100
    assert all(same[:9]) and not any(same[10:49]) and same[49] and not same[50] and same[99]


def test_time_limit_is_not_stored_as_the_end_of_the_episode():
    agent, _ = train(TinyEnv(), 4, GAMMA, list(range(4)), {**TINY, "learn_start": 10})
    assert len(agent.buffer) == 200
    assert agent.buffer.terminated.sum() == 0


@pytest.mark.parametrize("double", [False, True])
def test_dqn_finds_the_known_answer(double):
    # with a constant learning rate and noisy rewards the values keep moving a little: in 12 trial runs every
    # value was within 6% of the right answer. A one week discount instead of gamma^3 would be 23% off
    agent, _ = train(TinyEnv(), 300, GAMMA, list(range(300)), {**TINY, "double": double})
    q = np.array([agent.q_values(np.eye(2, dtype=np.float32)[s]) for s in range(2)])
    q_star = tiny_q_star(GAMMA)
    assert [network_policy(agent.online)(np.eye(2, dtype=np.float32)[s]) for s in range(2)] == [1, 0]
    assert np.allclose(q, q_star, rtol=0.1)


def test_training_is_reproducible():
    a = train(TinyEnv(), 30, GAMMA, list(range(30)), TINY, seed=3)[0]
    b = train(TinyEnv(), 30, GAMMA, list(range(30)), TINY, seed=3)[0]
    c = train(TinyEnv(), 30, GAMMA, list(range(30)), TINY, seed=4)[0]
    assert all(torch.equal(x, y) for x, y in zip(a.online.parameters(), b.online.parameters()))
    assert not all(torch.equal(x, y) for x, y in zip(a.online.parameters(), c.online.parameters()))


def test_learning_curve_is_recorded():
    costs = iter([5.0, 3.0, 4.0])
    seen = []

    def evaluate(net):
        seen.append(net)
        return next(costs)

    agent, curve = train(TinyEnv(), 30, GAMMA, list(range(30)), TINY, eval_every=10, evaluate=evaluate)
    assert curve == [(10, 500, 5.0), (20, 1000, 3.0), (30, 1500, 4.0)]
    assert all(net is agent.online for net in seen)

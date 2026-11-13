import gymnasium as gym
import numpy as np

from src import config
from src.policies import VIB_SCALE
from src.pump_env import CONTINUE, MINOR, PumpEnv
from src.tabular import (EST_EDGES, N_STATES, SHAPE, collect, discretise, epsilon, estimate_model, greedy_table,
                         q_learning, value_iteration)

GAMMA = 0.9


def make_obs(vib=0.0, eff=0.80, load=0.75):
    return np.array([vib / VIB_SCALE, 0.0, (0.80 - eff) / 0.08, (load - 0.60) / 0.30, 0.0, 0.0], dtype=np.float32)


def bands(obs):
    return np.unravel_index(discretise(obs), SHAPE)


class TinyEnv:
    # two states. In state 0: action 0 gives 1 and stays, action 1 gives 0 and moves to state 1 but takes
    # 3 weeks. In state 1: action 0 gives 2 and stays, action 1 gives 0 and goes back to state 0. Every
    # reward also gets noise with mean 0, so the right answer is the same but a constant learning rate
    # cannot settle on it
    action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None):
        self.s, self.t = 0, 0
        self.rng = np.random.default_rng(seed)
        return self.s, {}

    def step(self, a):
        reward = [[1.0, 0.0], [2.0, 0.0]][self.s][a] + self.rng.normal(0, 1.0)
        weeks = 3 if (self.s, a) == (0, 1) else 1
        self.s = self.s if a == 0 else 1 - self.s
        self.t += 1
        return self.s, reward, False, self.t >= 50, {"weeks": weeks}


def tiny_model(gamma):
    R = np.array([[1.0, 0.0], [2.0, 0.0]])
    M = np.zeros((2, 2, 2))
    M[0, 0, 0] = gamma
    M[0, 1, 1] = gamma**3
    M[1, 0, 1] = gamma
    M[1, 1, 0] = gamma
    return R, M


def test_bands_in_physical_units():
    assert bands(make_obs(vib=0.5))[0] == 0
    assert bands(make_obs(vib=4.9))[0] == 4
    assert bands(make_obs(vib=5.1))[0] == 5
    assert bands(make_obs(vib=6.5))[0] == 6
    assert bands(make_obs(eff=0.795))[1] == 0
    assert bands(make_obs(eff=0.745))[1] == 5
    assert bands(make_obs(eff=0.735))[1] == 6
    assert [bands(make_obs(load=x))[2] for x in config.LOAD_LEVELS] == [0, 1, 2]
    assert np.isclose(EST_EDGES[-1], 0.75)


def test_every_band_combination_has_its_own_state():
    seen = set()
    for vib in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]:
        for eff in [0.795, 0.785, 0.775, 0.765, 0.755, 0.745, 0.735]:
            for load in config.LOAD_LEVELS:
                seen.add(discretise(make_obs(vib, eff, load)))
    assert seen == set(range(N_STATES))


def test_value_iteration_on_a_known_mdp():
    R, M = tiny_model(GAMMA)
    V, Q, iterations = value_iteration(R, M, np.ones((2, 2), bool))
    v1 = 2 / (1 - GAMMA)
    v0 = max(1 / (1 - GAMMA), GAMMA**3 * v1)
    assert np.allclose(V, [v0, v1], atol=1e-6)
    assert list(Q.argmax(axis=1)) == [1, 0]
    assert np.isclose(Q[0, 1], GAMMA**3 * v1, atol=1e-6)
    assert iterations > 10
    assert np.max(np.abs(V - np.max(R + M @ V, axis=1))) < 1e-7


def test_unknown_actions_are_never_chosen():
    R, M = tiny_model(GAMMA)
    known = np.array([[True, False], [True, True]])
    V, Q, _ = value_iteration(R, M, known)
    assert np.isclose(V[0], 1 / (1 - GAMMA), atol=1e-6)
    assert list(greedy_table(Q, known)) == [0, 0]
    assert greedy_table(np.zeros((2, 3)), np.array([[False] * 3, [False, True, False]]))[0] == CONTINUE


def test_estimate_model_on_made_up_data():
    data = {"s": np.array([0, 0, 0, 1]), "a": np.array([1, 1, 1, 0]), "r": np.array([-1.0, -3.0, -2.0, 0.5]),
            "weeks": np.array([1, 2, 1, 1]), "s2": np.array([1, 1, 0, 1])}
    R, M, counts = estimate_model(data, GAMMA, n_states=2, n_actions=2)
    assert counts[0, 1] == 3 and counts[1, 0] == 1 and counts[0, 0] == 0
    assert np.isclose(R[0, 1], -2.0) and np.isclose(R[1, 0], 0.5)
    assert np.isclose(M[0, 1, 1], (GAMMA + GAMMA**2) / 3)
    assert np.isclose(M[0, 1, 0], GAMMA / 3)
    assert np.isclose(M[1, 0, 1], GAMMA)
    assert np.all(M[0, 0] == 0)


def test_q_learning_finds_the_known_answer():
    # with 3000 episodes every Q value is within 1% of the value iteration answer (0.2% when this was written)
    seeds = list(range(3000))
    Q, visits, _ = q_learning(TinyEnv(), 3000, GAMMA, seeds, state=lambda s: s, n_states=2)
    _, Q_star, _ = value_iteration(*tiny_model(GAMMA), np.ones((2, 2), bool))
    assert list(greedy_table(Q, visits > 0)) == [1, 0]
    assert np.allclose(Q, Q_star, rtol=0.01)


def test_q_learning_is_reproducible():
    seeds = list(range(50))
    a = q_learning(TinyEnv(), 50, GAMMA, seeds, state=lambda s: s, n_states=2, seed=4)[0]
    b = q_learning(TinyEnv(), 50, GAMMA, seeds, state=lambda s: s, n_states=2, seed=4)[0]
    assert np.array_equal(a, b)


def test_epsilon_schedule():
    assert epsilon(0, 100) == 1.0
    assert np.isclose(epsilon(25, 100), 0.51)
    assert np.isclose(epsilon(50, 100), 0.02) and epsilon(99, 100) == 0.02


def test_collect_gives_a_consistent_chain():
    env = PumpEnv(weeks=40)
    actions = iter([CONTINUE] * 10 + [MINOR] + [CONTINUE] * 100)
    data = collect(env, [3], lambda: lambda obs: next(actions))
    assert data["weeks"].sum() == 40
    assert np.array_equal(data["s"][1:], data["s2"][:-1])
    assert data["a"][10] == MINOR and np.isclose(data["r"][10], -(0.5 + 0.5 * 3 / 7), atol=0.05)

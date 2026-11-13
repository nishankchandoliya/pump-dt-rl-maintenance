import numpy as np

from src import config
from src.pump_env import CONTINUE, obs_scales

VIB_SCALE, _ = obs_scales(config.PUMP)

# band edges in physical units: excess vibration (mm/s) and 8 week efficiency. Temperature is left out:
# it only rises after vibration is already high, so in a table it would mostly split rare states (D39)
VIB_EDGES = [1, 2, 3, 4, 5, 6]
EFF_EDGES = [0.79, 0.78, 0.77, 0.76, 0.75, 0.74]
# the same efficiency edges written as values of the impeller estimate in the observation
EST_EDGES = [(config.PUMP["eff_new"] - e) / config.PUMP["eff_loss"] for e in EFF_EDGES]
SHAPE = (len(VIB_EDGES) + 1, len(EFF_EDGES) + 1, len(config.LOAD_LEVELS))
N_STATES = int(np.prod(SHAPE))
N_ACTIONS = 3

# settings of Q-learning: epsilon falls linearly to its end value over the first half of training
# and the learning rate of each state and action is 1 / (visits ^ ALPHA_POWER)
EPS_START, EPS_END, EPS_DECAY_SHARE = 1.0, 0.02, 0.5
ALPHA_POWER = 0.7


def discretise(obs):
    vib = np.searchsorted(VIB_EDGES, obs[0] * VIB_SCALE, side="right")
    eff = np.searchsorted(EST_EDGES, obs[2], side="right")
    load = int(round(obs[3] * (len(config.LOAD_LEVELS) - 1)))
    return int(np.ravel_multi_index((vib, eff, load), SHAPE))


def collect(env, seeds, behaviour, state=discretise):
    # one row per step: state, action, reward, weeks used by the step, next state
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        policy = behaviour()
        s = state(obs)
        truncated = False
        while not truncated:
            a = policy(obs)
            obs, r, _, truncated, info = env.step(a)
            s2 = state(obs)
            rows.append((s, a, r, info["weeks"], s2))
            s = s2
    s, a, r, weeks, s2 = (np.array(col) for col in zip(*rows))
    return {"s": s.astype(int), "a": a.astype(int), "r": r, "weeks": weeks, "s2": s2.astype(int)}


def estimate_model(data, gamma, n_states=N_STATES, n_actions=N_ACTIONS):
    # returns the mean reward R[s, a] and M[s, a, s2]: the mean of gamma^weeks over the steps from (s, a)
    # that ended in s2. Then Q(s, a) = R[s, a] + sum over s2 of M[s, a, s2] V(s2)
    s, a, s2 = data["s"], data["a"], data["s2"]
    counts = np.zeros((n_states, n_actions))
    R = np.zeros((n_states, n_actions))
    M = np.zeros((n_states, n_actions, n_states))
    np.add.at(counts, (s, a), 1)
    np.add.at(R, (s, a), data["r"])
    np.add.at(M, (s, a, s2), gamma ** data["weeks"])
    seen = counts > 0
    R[seen] /= counts[seen]
    M[seen] /= counts[seen][:, None]
    return R, M, counts


def value_iteration(R, M, known, tol=1e-8, max_iter=100_000):
    V = np.zeros(R.shape[0])
    for i in range(1, max_iter + 1):
        Q = np.where(known, R + M @ V, -np.inf)
        V_new = np.where(known.any(axis=1), Q.max(axis=1), 0.0)
        change = np.max(np.abs(V_new - V))
        V = V_new
        if change < tol:
            return V, Q, i
    raise RuntimeError(f"value iteration did not converge in {max_iter} iterations")


def greedy_table(Q, known):
    # best known action in each state; continue where nothing is known
    table = np.where(known, Q, -np.inf).argmax(axis=1)
    table[~known.any(axis=1)] = CONTINUE
    return table


def table_policy(table, state=discretise):
    return lambda obs: int(table[state(obs)])


def epsilon(episode, episodes):
    fraction = episode / (EPS_DECAY_SHARE * episodes)
    return max(EPS_END, EPS_START - (EPS_START - EPS_END) * fraction)


def q_learning(env, episodes, gamma, seeds, state=discretise, n_states=N_STATES, seed=0, eval_every=0, evaluate=None):
    rng = np.random.default_rng(seed)
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    visits = np.zeros((n_states, n_actions))
    curve = []
    for episode in range(episodes):
        eps = epsilon(episode, episodes)
        obs, _ = env.reset(seed=seeds[episode])
        s = state(obs)
        truncated = False
        while not truncated:
            a = int(rng.integers(n_actions)) if rng.random() < eps else int(Q[s].argmax())
            obs, r, terminated, truncated, info = env.step(a)
            s2 = state(obs)
            visits[s, a] += 1
            alpha = 1 / visits[s, a] ** ALPHA_POWER
            # a step of k weeks is discounted by gamma^k; at the time limit we still bootstrap (D06, D32)
            target = r if terminated else r + gamma ** info["weeks"] * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            truncated = truncated or terminated
        if eval_every and (episode + 1) % eval_every == 0:
            curve.append((episode + 1, evaluate(Q, visits)))
    return Q, visits, curve

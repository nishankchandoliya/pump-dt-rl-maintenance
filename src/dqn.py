import copy

import numpy as np
import torch
from torch import nn

# settings of the DQN agent, chosen in Phase 7 (D42 to D44)
SETTINGS = {
    "hidden": 64,
    "lr_start": 5e-4,
    "lr_end": 5e-5,
    "lr_decay_steps": 400_000,
    "batch_size": 64,
    "buffer_size": 200_000,
    "learn_start": 5_000,
    "train_every": 1,
    "target_every": 500,
    "eps_start": 1.0,
    "eps_end": 0.02,
    "eps_decay_steps": 100_000,
    "double": False,
}


def make_network(n_inputs, n_actions, hidden):
    return nn.Sequential(nn.Linear(n_inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, n_actions))


class ReplayBuffer:
    # stores the last `size` steps and gives random batches of them. When it is full the oldest step is overwritten
    def __init__(self, size, n_inputs, seed=0):
        self.obs = np.zeros((size, n_inputs), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.weeks = np.zeros(size, dtype=np.float32)
        self.next_obs = np.zeros((size, n_inputs), dtype=np.float32)
        self.terminated = np.zeros(size, dtype=np.float32)
        self.size = size
        self.pos = 0
        self.count = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, weeks, next_obs, terminated):
        i = self.pos
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.weeks[i], self.next_obs[i], self.terminated[i] = weeks, next_obs, terminated
        self.pos = (self.pos + 1) % self.size
        self.count = min(self.count + 1, self.size)

    def __len__(self):
        return self.count

    def sample(self, batch_size):
        idx = self.rng.integers(self.count, size=batch_size)
        columns = (self.obs, self.actions, self.rewards, self.weeks, self.next_obs, self.terminated)
        return [torch.as_tensor(c[idx]) for c in columns]


def td_targets(online, target, rewards, weeks, next_obs, terminated, gamma, double):
    # y = r + gamma^weeks * Q_target(s2, best action). At the time limit we still bootstrap: only a true end
    # of the episode (terminated) cuts the future off (D06, D32). Double DQN picks the best action with the
    # online network and takes its value from the target network
    with torch.no_grad():
        q_next = target(next_obs)
        if double:
            best = online(next_obs).argmax(dim=1, keepdim=True)
            next_value = q_next.gather(1, best).squeeze(1)
        else:
            next_value = q_next.max(dim=1).values
        return rewards + gamma**weeks * (1 - terminated) * next_value


def linear(start, end, step, steps):
    # moves from start to end in a straight line over `steps` steps, then stays at end
    return start + min(1.0, step / steps) * (end - start)


def epsilon(step, settings):
    return linear(settings["eps_start"], settings["eps_end"], step, settings["eps_decay_steps"])


class DQNAgent:
    def __init__(self, n_inputs, n_actions, gamma, settings=SETTINGS, seed=0):
        torch.manual_seed(seed)
        self.s = dict(settings)
        self.gamma = gamma
        self.n_actions = n_actions
        self.online = make_network(n_inputs, n_actions, self.s["hidden"])
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.s["lr_start"])
        self.buffer = ReplayBuffer(self.s["buffer_size"], n_inputs, seed)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    def q_values(self, obs):
        with torch.no_grad():
            return self.online(torch.as_tensor(obs, dtype=torch.float32)).numpy()

    def act(self, obs, eps=0.0):
        if self.rng.random() < eps:
            return int(self.rng.integers(self.n_actions))
        return int(self.q_values(obs).argmax())

    def update(self):
        # the learning rate falls during training, so the last network settles (D43)
        for group in self.optimizer.param_groups:
            group["lr"] = linear(self.s["lr_start"], self.s["lr_end"], self.steps, self.s["lr_decay_steps"])
        obs, actions, rewards, weeks, next_obs, terminated = self.buffer.sample(self.s["batch_size"])
        y = td_targets(self.online, self.target, rewards, weeks, next_obs, terminated, self.gamma, self.s["double"])
        q = self.online(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def observe(self, obs, action, reward, weeks, next_obs, terminated):
        # store one step, then train and copy the target network on their schedules
        self.buffer.add(obs, action, reward, weeks, next_obs, terminated)
        self.steps += 1
        if self.steps >= self.s["learn_start"] and self.steps % self.s["train_every"] == 0:
            self.update()
        if self.steps % self.s["target_every"] == 0:
            self.target.load_state_dict(self.online.state_dict())


def load_network(path):
    # the sizes of the layers are read from the saved weights: "0.weight" is the first layer, "4.weight" the last
    saved = torch.load(path)
    state = saved["state"]
    net = make_network(state["0.weight"].shape[1], state["4.weight"].shape[0], saved["settings"]["hidden"])
    net.load_state_dict(state)
    return net


def network_policy(net):
    def policy(obs):
        with torch.no_grad():
            return int(net(torch.as_tensor(obs, dtype=torch.float32)).argmax())
    return policy


def train(env, episodes, gamma, seeds, settings=SETTINGS, seed=0, eval_every=0, evaluate=None):
    # evaluate(net) gives the mean cost of the greedy policy; it is only recorded for the learning curve
    torch.set_num_threads(1)
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, gamma, settings, seed)
    curve = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seeds[episode])
        done = False
        while not done:
            action = agent.act(obs, epsilon(agent.steps, agent.s))
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.observe(obs, action, reward, info["weeks"], next_obs, terminated)
            obs = next_obs
            done = terminated or truncated
        if eval_every and (episode + 1) % eval_every == 0:
            curve.append((episode + 1, agent.steps, evaluate(agent.online)))
    return agent, curve

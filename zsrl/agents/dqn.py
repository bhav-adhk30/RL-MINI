import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from zsrl.agents.base import Agent


class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        # state_dim is 2082 now that FrozenEncoder returns 2x2x512 spatial
        # features instead of one globally-pooled 512-vector (see
        # zsrl/encoder.py). Reverted from the 128-64 shrink used for the
        # memorisation fix back to 512-256: capacity should track the
        # richer input, and augmentation is already handling overfitting.
        # See results/spatial_pooling_notes.md.
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity, state_dim, n_actions):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, state_dim), dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.m2 = np.zeros((capacity, n_actions), dtype=np.float32)
        self.idx, self.full = 0, False

    def add(self, s, a, r, s2, done, next_mask):
        i = self.idx
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.s2[i], self.d[i] = s2, float(done)
        self.m2[i] = next_mask.astype(np.float32)
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size, rng):
        ids = rng.integers(len(self), size=batch_size)
        return (self.s[ids], self.a[ids], self.r[ids],
                self.s2[ids], self.d[ids], self.m2[ids])


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, state_dim, n_actions, device,
                 lr=1e-4, gamma=0.95, buffer_size=100_000, batch_size=64,
                 target_sync=500, train_every=2, learn_start=1000,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=20_000, seed=0):
        self.device, self.n_actions = device, n_actions
        self.gamma, self.batch_size = gamma, batch_size
        self.target_sync, self.train_every = target_sync, train_every
        self.learn_start = learn_start
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps

        self.online = QNetwork(state_dim, n_actions).to(device)
        self.target = QNetwork(state_dim, n_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, state_dim, n_actions)
        self.rng = np.random.default_rng(seed)
        self.steps = 0

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def q_values(self, state, mask=None):
        with torch.no_grad():
            x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q = self.online(x).squeeze(0).cpu().numpy()
        if mask is not None:
            q = np.where(mask, q, -np.inf)
        return q

    def act(self, state, mask, greedy=False):
        legal = np.flatnonzero(mask)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.choice(legal))
        return int(np.argmax(self.q_values(state, mask)))

    def observe(self, s, a, r, s2, done, next_mask):
        self.buffer.add(s, a, r, s2, done, next_mask)
        self.steps += 1

    def update(self):
        if len(self.buffer) < self.learn_start:
            return None
        if self.steps % self.train_every != 0:
            return None

        s, a, r, s2, d, m2 = self.buffer.sample(self.batch_size, self.rng)
        s = torch.from_numpy(s).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)
        d = torch.from_numpy(d).to(self.device)
        m2 = torch.from_numpy(m2).to(self.device).bool()

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            NEG = torch.finfo(torch.float32).min
            q_next_online = self.online(s2).masked_fill(~m2, NEG)
            next_a = q_next_online.argmax(dim=1, keepdim=True)
            q_next_target = self.target(s2).masked_fill(~m2, NEG)
            next_q = q_next_target.gather(1, next_a).squeeze(1)
            next_q = torch.nan_to_num(next_q, neginf=0.0)
            y = r + self.gamma * (1.0 - d) * next_q

        loss = F.smooth_l1_loss(q, y)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        if self.steps % self.target_sync == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def save(self, path):
        torch.save({"online": self.online.state_dict(), "steps": self.steps}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ck["online"])
        self.target.load_state_dict(ck["online"])
        self.steps = ck.get("steps", 0)

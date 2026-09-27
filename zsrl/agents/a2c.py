import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from zsrl.agents.base import Agent

NEG_INF = -1e9


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.actor_head = nn.Linear(256, n_actions)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)


class A2CAgent(Agent):
    name = "a2c"

    def __init__(self, state_dim, n_actions, device,
                 lr=3e-4, gamma=0.95, batch_size=64,
                 value_coef=0.5, entropy_coef=0.02, grad_clip=0.5, seed=0):
        self.device, self.n_actions = device, n_actions
        self.gamma, self.batch_size = gamma, batch_size
        self.value_coef, self.entropy_coef, self.grad_clip = value_coef, entropy_coef, grad_clip

        self.net = ActorCriticNetwork(state_dim, n_actions).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.rng = np.random.default_rng(seed)

        self._buffer = []
        self._policy_entropy = 0.0
        self._last_mask = None

    def _masked_logits(self, logits, mask):
        return logits.masked_fill(~mask, NEG_INF)

    def q_values(self, state, mask=None):
        """Action probabilities (not literal Q-values) for logging/the app."""
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(x)
            if mask is not None:
                m = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
                logits = self._masked_logits(logits, m)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs

    def act(self, state, mask, greedy=False):
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        m = torch.from_numpy(mask).bool().unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, _ = self.net(x)
            logits = self._masked_logits(logits, m)
            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
        self._last_mask = mask.copy()
        return int(action.item())

    def observe(self, s, a, r, s2, done, next_mask):
        self._buffer.append({"s": s, "a": a, "r": r, "s2": s2, "done": done, "mask": self._last_mask})

    def update(self):
        if len(self._buffer) < self.batch_size:
            return None

        batch, self._buffer = self._buffer, []

        states = torch.from_numpy(np.stack([t["s"] for t in batch])).float().to(self.device)
        actions = torch.tensor([t["a"] for t in batch], dtype=torch.long, device=self.device)
        masks = torch.from_numpy(np.stack([t["mask"] for t in batch])).bool().to(self.device)
        rewards = [t["r"] for t in batch]
        dones = [bool(t["done"]) for t in batch]
        last_s2 = torch.from_numpy(batch[-1]["s2"]).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            bootstrap = 0.0 if dones[-1] else self.net(last_s2)[1].item()

        returns = [0.0] * len(batch)
        running = bootstrap
        for i in reversed(range(len(batch))):
            running = rewards[i] + self.gamma * running * (1.0 - float(dones[i]))
            returns[i] = running
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        logits, values = self.net(states)
        logits = self._masked_logits(logits, masks)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        advantages = (returns_t - values).detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns_t)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.opt.step()

        self._policy_entropy = float(entropy.item())
        return float(loss.item())

    @property
    def policy_entropy(self):
        return self._policy_entropy

    def save(self, path):
        torch.save({"model": self.net.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["model"])

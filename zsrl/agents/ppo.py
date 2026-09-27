import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from zsrl.agents.base import Agent


class ActorCritic(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.actor(h), self.critic(h).squeeze(-1)


class RolloutBuffer:
    def __init__(self, capacity, state_dim, n_actions):
        self.capacity = capacity
        self.s = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.mask = np.zeros((capacity, n_actions), dtype=np.bool_)
        self.logp = np.zeros(capacity, dtype=np.float32)
        self.val = np.zeros(capacity, dtype=np.float32)
        self.idx = 0

    def add(self, s, a, r, done, mask, logp, val):
        i = self.idx
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.d[i] = float(done)
        self.mask[i] = mask
        self.logp[i] = logp
        self.val[i] = val
        self.idx += 1

    def full(self):
        return self.idx >= self.capacity

    def clear(self):
        self.idx = 0


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, state_dim, n_actions, device,
                 lr=3e-4, rollout_len=2048, epochs=4, minibatch_size=64,
                 clip_eps=0.2, gamma=0.95, gae_lambda=0.95,
                 vf_coef=0.5, ent_coef=0.01, grad_clip=0.5, seed=0):
        self.device, self.n_actions = device, n_actions
        self.rollout_len = rollout_len
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.grad_clip = grad_clip

        self.net = ActorCritic(state_dim, n_actions).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.buffer = RolloutBuffer(rollout_len, state_dim, n_actions)
        self.rng = np.random.default_rng(seed)

        # cached between act() and observe() -- interface separates them
        self._last_logp = None
        self._last_val = None
        self._last_mask = None
        # cached so _learn() can bootstrap the value of the state
        # just past the end of the buffer, instead of assuming 0
        self._last_next_state = None
        self._last_done = False

        self._policy_entropy = 0.0

    @property
    def policy_entropy(self):
        return self._policy_entropy

    def _logits_and_value(self, state, mask):
        x = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        NEG = torch.finfo(torch.float32).min
        m = torch.from_numpy(mask).to(self.device).unsqueeze(0)
        logits = logits.masked_fill(~m, NEG)
        return logits.squeeze(0), value.squeeze(0)

    def act(self, state, mask, greedy=False):
        with torch.no_grad():
            logits, value = self._logits_and_value(state, mask)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits) if greedy else dist.sample()
            logp = dist.log_prob(action)

        # stash for observe() -- act() and observe() are separate calls
        self._last_logp = float(logp.item())
        self._last_val = float(value.item())
        self._last_mask = mask.copy()
        return int(action.item())

    def observe(self, s, a, r, s2, done, next_mask):
        self.buffer.add(s, a, r, done, self._last_mask,
                         self._last_logp, self._last_val)
        self._last_next_state = s2
        self._last_done = done

    def update(self):
        if not self.buffer.full():
            return None
        return self._learn()

    def _learn(self):
        buf = self.buffer
        n = buf.idx

        with torch.no_grad():
            if self._last_done:
                last_val = 0.0
            else:
                x = torch.from_numpy(self._last_next_state).float()
                x = x.unsqueeze(0).to(self.device)
                _, v = self.net(x)
                last_val = float(v.item())

        values = buf.val
        rewards, dones = buf.r, buf.d

        advantages = np.zeros(n, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(n)):
            next_val = values[t + 1] if t + 1 < n else last_val
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * next_nonterminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
            advantages[t] = lastgaelam
        returns = advantages + values

        states = torch.from_numpy(buf.s).float().to(self.device)
        actions = torch.from_numpy(buf.a).to(self.device)
        old_logp = torch.from_numpy(buf.logp).float().to(self.device)
        masks = torch.from_numpy(buf.mask).to(self.device)
        adv_t = torch.from_numpy(advantages).float().to(self.device)
        ret_t = torch.from_numpy(returns).float().to(self.device)

        NEG = torch.finfo(torch.float32).min
        idxs = np.arange(n)
        losses, entropies = [], []

        for _ in range(self.epochs):
            self.rng.shuffle(idxs)
            for start in range(0, n, self.minibatch_size):
                mb = torch.from_numpy(idxs[start:start + self.minibatch_size]).long().to(self.device)

                logits, values_pred = self.net(states[mb])
                logits = logits.masked_fill(~masks[mb], NEG)
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(actions[mb])
                entropy = dist.entropy().mean()

                mb_adv = adv_t[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                ratio = torch.exp(new_logp - old_logp[mb])
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values_pred, ret_t[mb])

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.opt.step()

                losses.append(float(loss.item()))
                entropies.append(float(entropy.item()))

        self._policy_entropy = float(np.mean(entropies))
        self.buffer.clear()
        return float(np.mean(losses))

    def q_values(self, state, mask=None):
        if mask is None:
            mask = np.ones(self.n_actions, dtype=bool)
        with torch.no_grad():
            logits, _ = self._logits_and_value(state, mask)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def save(self, path):
        torch.save({"net": self.net.state_dict()}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ck["net"])

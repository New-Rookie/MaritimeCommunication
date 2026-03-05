"""
MAPPO baseline (without GCN) for link selection.
Centralised critic + decentralised actors.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple


class MAPPOActor(nn.Module):
    def __init__(self, obs_dim: int, max_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, max_actions),
        )

    def forward(self, x):
        return self.net(x)


class MAPPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MAPPOAgent:
    def __init__(self, obs_dim: int = 15, max_actions: int = 15,
                 state_dim: int = 64, lr: float = 3e-4,
                 gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, epochs: int = 4,
                 batch_size: int = 64, device: str = "auto"):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else device if device != "auto" else "cpu"
        )
        self.actor = MAPPOActor(obs_dim, max_actions).to(self.device)
        self.critic = MAPPOCritic(state_dim).to(self.device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr
        )
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_actions = max_actions
        self._buffer: List = []

    def select_action(self, obs: np.ndarray,
                      valid_mask: np.ndarray) -> Tuple[int, float]:
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        logits = self.actor(x).squeeze(0)
        mask = torch.FloatTensor(valid_mask).to(self.device)
        logits = logits + (mask - 1) * 1e9
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a.item(), dist.log_prob(a).item()

    def get_value(self, state: np.ndarray) -> float:
        x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        return self.critic(x).item()

    def store(self, obs, action, reward, log_prob, value, done):
        self._buffer.append((obs, action, reward, log_prob, value, done))

    def update(self) -> float:
        if len(self._buffer) < 2:
            return 0.0
        rewards = np.array([t[2] for t in self._buffer], dtype=np.float32)
        old_lps = np.array([t[3] for t in self._buffer], dtype=np.float32)
        values = np.array([t[4] for t in self._buffer], dtype=np.float32)
        dones = np.array([t[5] for t in self._buffer], dtype=np.float32)

        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nv = 0.0 if t == len(rewards) - 1 else values[t + 1]
            d = rewards[t] + self.gamma * nv * (1 - dones[t]) - values[t]
            gae = d + self.gamma * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
        rets = adv + values

        obs_t = torch.FloatTensor(np.array([t[0] for t in self._buffer])).to(self.device)
        act_t = torch.LongTensor([t[1] for t in self._buffer]).to(self.device)
        old_lp_t = torch.FloatTensor(old_lps).to(self.device)
        adv_t = torch.FloatTensor(adv).to(self.device)
        ret_t = torch.FloatTensor(rets).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        n = len(self._buffer)
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, self.batch_size):
                e = min(s + self.batch_size, n)
                mb = idx[s:e]
                logits = self.actor(obs_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act_t[mb])
                ent = dist.entropy()
                ratio = torch.exp(lp - old_lp_t[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - self.clip_eps,
                                 1 + self.clip_eps) * adv_t[mb]
                loss = -torch.min(s1, s2).mean() - 0.01 * ent.mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        self._buffer.clear()
        return total_loss

"""
PPO (Proximal Policy Optimisation) – single-agent, PyTorch + CUDA.

Used as a baseline in P1 experiments (centralised single-policy PPO).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def act(self, obs: np.ndarray, device: torch.device):
        x = torch.FloatTensor(obs).unsqueeze(0).to(device)
        logits, value = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, values = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(-1), entropy


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net = ActorCritic(obs_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size

        self._buffer: List = []

    def select_action(self, obs: np.ndarray) -> Tuple[int, float, float]:
        return self.net.act(obs, self.device)

    def store(self, obs, action, reward, log_prob, value, done):
        self._buffer.append((obs, action, reward, log_prob, value, done))

    def update(self) -> float:
        if len(self._buffer) < 2:
            return 0.0
        obs_arr = np.array([t[0] for t in self._buffer], dtype=np.float32)
        act_arr = np.array([t[1] for t in self._buffer])
        rew_arr = np.array([t[2] for t in self._buffer], dtype=np.float32)
        old_lp = np.array([t[3] for t in self._buffer], dtype=np.float32)
        val_arr = np.array([t[4] for t in self._buffer], dtype=np.float32)
        done_arr = np.array([t[5] for t in self._buffer], dtype=np.float32)

        # GAE
        advantages = np.zeros_like(rew_arr)
        gae = 0.0
        for t in reversed(range(len(rew_arr))):
            next_val = 0.0 if t == len(rew_arr) - 1 else val_arr[t + 1]
            delta = rew_arr[t] + self.gamma * next_val * (1 - done_arr[t]) - val_arr[t]
            gae = delta + self.gamma * self.lam * (1 - done_arr[t]) * gae
            advantages[t] = gae
        returns = advantages + val_arr

        obs_t = torch.FloatTensor(obs_arr).to(self.device)
        act_t = torch.LongTensor(act_arr).to(self.device)
        old_lp_t = torch.FloatTensor(old_lp).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        n = len(obs_arr)
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                mb = idx[start:end]
                lp, vals, ent = self.net.evaluate(obs_t[mb], act_t[mb])
                ratio = torch.exp(lp - old_lp_t[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv_t[mb]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = 0.5 * (ret_t[mb] - vals).pow(2).mean()
                entropy_bonus = -0.01 * ent.mean()
                loss = actor_loss + critic_loss + entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        self._buffer.clear()
        return total_loss

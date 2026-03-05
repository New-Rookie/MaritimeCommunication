"""
Standard MATD3 (Multi-Agent TD3) baseline for resource management.
Uses standard MLP (no temporal architecture, no hierarchical replay).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class _Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class _Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


class MATD3Agent:
    def __init__(self, obs_dim=12, action_dim=4, state_dim=64,
                 lr_a=1e-4, lr_c=1e-3, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 buffer_size=100000, batch_size=128, device="auto"):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else device if device != "auto" else "cpu"
        )
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size

        self.actor = _Actor(obs_dim, action_dim).to(self.device)
        self.actor_target = _Actor(obs_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = _Critic(state_dim, action_dim).to(self.device)
        self.critic_target = _Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.replay = deque(maxlen=buffer_size)
        self._update_count = 0

    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            a = self.actor(x).cpu().numpy()[0]
        if explore:
            a += np.random.normal(0, 0.1, size=a.shape)
        return np.clip(a, -1.0, 1.0)

    def store(self, obs, action, reward, next_obs, done, state, next_state):
        self.replay.append((obs, action, reward, next_obs, done, state, next_state))

    def update(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0
        batch = random.sample(self.replay, self.batch_size)
        obs = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        acts = torch.FloatTensor(np.array([t[1] for t in batch])).to(self.device)
        rews = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        nobs = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        states = torch.FloatTensor(np.array([t[5] for t in batch])).to(self.device)
        nstates = torch.FloatTensor(np.array([t[6] for t in batch])).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(acts) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            na = (self.actor_target(nobs) + noise).clamp(-1, 1)
            tq1, tq2 = self.critic_target(nstates, na)
            target = rews + self.gamma * (1 - dones) * torch.min(tq1, tq2)

        q1, q2 = self.critic(states, acts)
        c_loss = ((q1 - target)**2).mean() + ((q2 - target)**2).mean()
        self.opt_c.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_c.step()

        self._update_count += 1
        if self._update_count % self.policy_delay == 0:
            a_loss = -self.critic.q1(
                torch.cat([states, self.actor(obs)], dim=-1)
            ).mean()
            self.opt_a.zero_grad()
            a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt_a.step()
            for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return c_loss.item()

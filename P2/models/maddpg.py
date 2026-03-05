"""
MADDPG baseline for link selection.
Multi-Agent DDPG with centralised critic.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from collections import deque
import random


class _Actor(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class _Critic(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action_onehot):
        return self.net(torch.cat([state, action_onehot], dim=-1)).squeeze(-1)


class MADDPGAgent:
    def __init__(self, obs_dim=15, n_actions=15, state_dim=64,
                 lr_a=1e-4, lr_c=1e-3, gamma=0.99, tau=0.01,
                 buffer_size=50000, batch_size=64, device="auto"):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available()
            else device if device != "auto" else "cpu"
        )
        self.actor = _Actor(obs_dim, n_actions).to(self.device)
        self.actor_target = _Actor(obs_dim, n_actions).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = _Critic(state_dim, n_actions).to(self.device)
        self.critic_target = _Critic(state_dim, n_actions).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_c)
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.replay = deque(maxlen=buffer_size)

    def select_action(self, obs: np.ndarray,
                      valid_mask: np.ndarray,
                      explore: bool = True) -> int:
        x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        probs = self.actor(x).squeeze(0).detach().cpu().numpy()
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.maximum(probs, 0.0) * valid_mask
        if probs.sum() < 1e-10:
            probs = valid_mask.copy()
        if probs.sum() < 1e-10:
            return 0
        probs = probs / probs.sum()
        probs = np.nan_to_num(probs, nan=0.0)
        if probs.sum() < 1e-10:
            return 0
        probs = probs / probs.sum()
        if explore:
            return int(np.random.choice(len(probs), p=probs))
        return int(np.argmax(probs))

    def store(self, obs, action, reward, next_obs, done, state, next_state):
        self.replay.append((obs, action, reward, next_obs, done, state, next_state))

    def update(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0
        batch = random.sample(self.replay, self.batch_size)
        obs = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        acts = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rews = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        nobs = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).to(self.device)
        states = torch.FloatTensor(np.array([t[5] for t in batch])).to(self.device)
        nstates = torch.FloatTensor(np.array([t[6] for t in batch])).to(self.device)

        act_oh = torch.zeros(self.batch_size, self.n_actions, device=self.device)
        act_oh.scatter_(1, acts.unsqueeze(1), 1.0)

        with torch.no_grad():
            na = self.actor_target(nobs)
            tq = self.critic_target(nstates, na)
            target = rews + self.gamma * (1 - dones) * tq

        q = self.critic(states, act_oh)
        critic_loss = ((q - target) ** 2).mean()
        self.opt_c.zero_grad()
        critic_loss.backward()
        self.opt_c.step()

        a_pred = self.actor(obs)
        actor_loss = -self.critic(states, a_pred).mean()
        self.opt_a.zero_grad()
        actor_loss.backward()
        self.opt_a.step()

        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item()

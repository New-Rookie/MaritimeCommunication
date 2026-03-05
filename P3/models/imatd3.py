"""
IMATD3 – Improved Multi-Agent TD3 for resource management.

Two key improvements over standard MATD3:

1. Hierarchical Experience Replay (HER):
   - Short-term buffer (10K, sampled 50%) for fast adaptation
   - Mid-term buffer (30K, high TD-error, sampled 30%) for important transitions
   - Long-term buffer (50K, periodic snapshots, sampled 20%) to prevent catastrophic forgetting

2. Causal Temporal Architecture:
   - 1D dilated causal convolutions over state history (window H=16)
   - Causal attention with mask (attend only to past)
   - Replaces MLP in both Actor and Critic
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple

HISTORY_LEN = 16


# ===================================================================
# Causal Temporal Architecture
# ===================================================================

class CausalConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self.padding, dilation=dilation)

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(x)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class CausalAttention(nn.Module):
    def __init__(self, dim, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.register_buffer("_mask", None)

    def _causal_mask(self, T, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return mask

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        mask = self._causal_mask(T, x.device)
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class CausalTemporalBlock(nn.Module):
    """Causal conv + causal attention fusion."""
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.conv1 = CausalConv1D(dim, hidden, kernel_size=3, dilation=1)
        self.conv2 = CausalConv1D(hidden, hidden, kernel_size=3, dilation=2)
        self.attn = CausalAttention(hidden, n_heads=4)
        self.proj = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(hidden)

    def forward(self, x):
        # x: (B, T, D)
        # Conv path
        xc = x.permute(0, 2, 1)  # (B, D, T)
        hc = self.relu(self.conv1(xc))
        hc = self.relu(self.conv2(hc))
        hc = hc.permute(0, 2, 1)  # (B, T, hidden)
        # Attention path
        ha = self.attn(hc)
        # Fusion
        out = self.ln(self.proj(hc + ha))
        return out


# ===================================================================
# Actor and Critic with Causal Temporal Architecture
# ===================================================================

class TemporalActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=64):
        super().__init__()
        self.temporal = CausalTemporalBlock(obs_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim), nn.Tanh(),
        )

    def forward(self, state_history):
        # state_history: (B, T, obs_dim)
        h = self.temporal(state_history)
        return self.head(h[:, -1, :])  # use last time step


class TemporalCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=64):
        super().__init__()
        self.temporal = CausalTemporalBlock(state_dim, hidden)
        self.q1 = nn.Sequential(
            nn.Linear(hidden + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(hidden + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state_history, action):
        h = self.temporal(state_history)[:, -1, :]
        sa = torch.cat([h, action], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


# ===================================================================
# Hierarchical Experience Replay
# ===================================================================

class HierarchicalReplayBuffer:
    def __init__(self, short_cap=10000, mid_cap=30000, long_cap=50000):
        self.short = deque(maxlen=short_cap)
        self.mid = deque(maxlen=mid_cap)
        self.long = deque(maxlen=long_cap)
        self._episode_buffer = []
        self._episode_rewards = []

    def add(self, transition, td_error: float = 0.0):
        self.short.append(transition)
        if abs(td_error) > 0.5:
            self.mid.append(transition)

    def end_episode(self, episode_reward: float):
        self._episode_rewards.append(episode_reward)
        if len(self._episode_rewards) > 200:
            self._episode_rewards = self._episode_rewards[-200:]
        if len(self._episode_rewards) >= 10:
            threshold = np.percentile(self._episode_rewards[-100:], 75)
            if episode_reward >= threshold:
                for t in self._episode_buffer[-50:]:
                    self.long.append(t)
        self._episode_buffer.clear()

    def store_for_episode(self, transition):
        self._episode_buffer.append(transition)

    def sample(self, batch_size: int):
        n_short = int(batch_size * 0.5)
        n_mid = int(batch_size * 0.3)
        n_long = batch_size - n_short - n_mid

        samples = []

        def _sample_from(buf, n):
            if len(buf) == 0:
                return []
            n = min(n, len(buf))
            indices = random.sample(range(len(buf)), n)
            return [buf[i] for i in indices]

        samples.extend(_sample_from(self.short, n_short))
        samples.extend(_sample_from(self.mid, n_mid))
        samples.extend(_sample_from(self.long, n_long))

        while len(samples) < batch_size and len(self.short) > 0:
            samples.extend(_sample_from(self.short,
                                        min(batch_size - len(samples),
                                            len(self.short))))
        return samples[:batch_size]

    def __len__(self):
        return len(self.short) + len(self.mid) + len(self.long)


# ===================================================================
# IMATD3 Agent
# ===================================================================

class IMATD3Agent:
    """Improved Multi-Agent TD3 with HER + Causal Temporal Architecture."""

    def __init__(
        self,
        obs_dim: int = 12,
        action_dim: int = 4,
        global_state_dim: int = 64,
        hidden: int = 64,
        lr_a: float = 1e-4,
        lr_c: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        batch_size: int = 128,
        device: str = "auto",
    ):
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

        self.actor = TemporalActor(obs_dim, action_dim, hidden).to(self.device)
        self.actor_target = TemporalActor(obs_dim, action_dim, hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TemporalCritic(global_state_dim, action_dim, hidden).to(self.device)
        self.critic_target = TemporalCritic(global_state_dim, action_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_a = optim.Adam(self.actor.parameters(), lr=lr_a)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=lr_c)

        self.replay = HierarchicalReplayBuffer()
        self._update_count = 0

        # State history buffer per agent
        self._histories: dict = {}

    def _get_history(self, agent_id: int, obs: np.ndarray) -> np.ndarray:
        if agent_id not in self._histories:
            self._histories[agent_id] = deque(
                [np.zeros_like(obs)] * HISTORY_LEN, maxlen=HISTORY_LEN
            )
        self._histories[agent_id].append(obs)
        return np.array(list(self._histories[agent_id]), dtype=np.float32)

    def select_action(self, obs: np.ndarray, agent_id: int = 0,
                      explore: bool = True) -> np.ndarray:
        history = self._get_history(agent_id, obs)
        h_t = torch.FloatTensor(history).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(h_t).cpu().numpy()[0]
        if explore:
            action += np.random.normal(0, 0.1, size=action.shape)
        return np.clip(action, -1.0, 1.0)

    def store(self, obs_hist, action, reward, next_obs_hist,
              done, state_hist, next_state_hist, td_error=0.0):
        transition = (obs_hist, action, reward, next_obs_hist,
                      done, state_hist, next_state_hist)
        self.replay.add(transition, td_error)
        self.replay.store_for_episode(transition)

    def end_episode(self, episode_reward: float):
        self.replay.end_episode(episode_reward)
        self.reset_histories()

    def update(self) -> float:
        if len(self.replay) < self.batch_size:
            return 0.0

        batch = self.replay.sample(self.batch_size)
        obs_h = torch.FloatTensor(np.array([t[0] for t in batch])).to(self.device)
        acts = torch.FloatTensor(np.array([t[1] for t in batch])).to(self.device)
        rews = torch.FloatTensor(np.array([t[2] for t in batch])).to(self.device)
        nobs_h = torch.FloatTensor(np.array([t[3] for t in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in batch])).to(self.device)
        state_h = torch.FloatTensor(np.array([t[5] for t in batch])).to(self.device)
        nstate_h = torch.FloatTensor(np.array([t[6] for t in batch])).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(acts) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(nobs_h) + noise).clamp(-1, 1)
            tq1, tq2 = self.critic_target(nstate_h, next_actions)
            target_q = rews + self.gamma * (1 - dones) * torch.min(tq1, tq2)

        q1, q2 = self.critic(state_h, acts)
        td_errors = (torch.min(q1, q2) - target_q).abs().detach().cpu().numpy()
        critic_loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()

        self.opt_c.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.opt_c.step()

        # Update mid-term buffer with high TD-error transitions
        for i, tr in enumerate(batch):
            if i < len(td_errors) and abs(td_errors[i]) > 0.5:
                self.replay.mid.append(tr)

        self._update_count += 1
        actor_loss_val = 0.0
        if self._update_count % self.policy_delay == 0:
            a_pred = self.actor(obs_h)
            q1_pred, _ = self.critic(state_h, a_pred)
            actor_loss = -q1_pred.mean()
            self.opt_a.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.opt_a.step()
            actor_loss_val = actor_loss.item()

            for tp, p in zip(self.actor_target.parameters(),
                             self.actor.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for tp, p in zip(self.critic_target.parameters(),
                             self.critic.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return critic_loss.item()

    def reset_histories(self):
        self._histories.clear()

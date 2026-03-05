"""
MAPPO + GCN for link selection.

Architecture:
  1. GCN encodes the network topology → per-node embeddings
  2. Each agent (data-source node) uses its embedding + local obs
     to select the next-hop from its current neighbours.
  3. Centralised critic sees global state; decentralised actors
     see only local information + GCN embedding.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict

from .gcn import GCNEncoder


class Actor(nn.Module):
    def __init__(self, embed_dim: int, local_dim: int, max_neighbours: int,
                 hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim + local_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, max_neighbours),
        )

    def forward(self, embed, local_obs):
        x = torch.cat([embed, local_obs], dim=-1)
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, global_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state):
        return self.net(global_state).squeeze(-1)


class MAPPOGCNAgent:
    """MAPPO + GCN agent for multi-agent link selection."""

    def __init__(
        self,
        n_node_features: int = 7,
        gcn_hidden: int = 64,
        gcn_out: int = 32,
        local_obs_dim: int = 8,
        max_neighbours: int = 15,
        global_state_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
        device: str = "auto",
    ):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            device if device != "auto" else "cpu"
        )
        self.gcn = GCNEncoder(n_node_features, gcn_hidden, gcn_out).to(self.device)
        self.actor = Actor(gcn_out, local_obs_dim, max_neighbours).to(self.device)
        self.critic = Critic(global_state_dim).to(self.device)

        params = (list(self.gcn.parameters()) +
                  list(self.actor.parameters()) +
                  list(self.critic.parameters()))
        self.optimizer = optim.Adam(params, lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_neighbours = max_neighbours
        self.gcn_out = gcn_out
        self._buffer: List = []

    def get_embeddings(self, node_features: np.ndarray,
                       edge_index: np.ndarray) -> torch.Tensor:
        x = torch.FloatTensor(node_features).to(self.device)
        ei = torch.LongTensor(edge_index).to(self.device)
        return self.gcn(x, ei)

    def select_action(self, embedding: torch.Tensor,
                      local_obs: np.ndarray,
                      valid_mask: np.ndarray) -> Tuple[int, float, float]:
        lo = torch.FloatTensor(local_obs).unsqueeze(0).to(self.device)
        emb = embedding.unsqueeze(0) if embedding.dim() == 1 else embedding
        logits = self.actor(emb, lo).squeeze(0)

        mask = torch.FloatTensor(valid_mask).to(self.device)
        logits = logits + (mask - 1) * 1e9

        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), 0.0

    def get_value(self, global_state: np.ndarray) -> float:
        gs = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        return self.critic(gs).item()

    def store(self, obs_dict, action, reward, log_prob, value, done):
        self._buffer.append((obs_dict, action, reward, log_prob, value, done))

    def update(self) -> float:
        if len(self._buffer) < 2:
            return 0.0

        rewards = np.array([t[2] for t in self._buffer], dtype=np.float32)
        old_lps = np.array([t[3] for t in self._buffer], dtype=np.float32)
        values = np.array([t[4] for t in self._buffer], dtype=np.float32)
        dones = np.array([t[5] for t in self._buffer], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nv = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * nv * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)
        old_lp_t = torch.FloatTensor(old_lps).to(self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        total_loss = 0.0
        n = len(self._buffer)
        act_arr = torch.LongTensor([t[1] for t in self._buffer]).to(self.device)

        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                mb = idx[start:end]

                mb_logits = []
                for i in mb:
                    od = self._buffer[i][0]
                    emb = torch.FloatTensor(od["embedding"]).unsqueeze(0).to(self.device)
                    lo = torch.FloatTensor(od["local_obs"]).unsqueeze(0).to(self.device)
                    lg = self.actor(emb, lo)
                    mask = torch.FloatTensor(od["mask"]).to(self.device)
                    lg = lg + (mask - 1) * 1e9
                    mb_logits.append(lg.squeeze(0))
                mb_logits = torch.stack(mb_logits)

                dist = torch.distributions.Categorical(logits=mb_logits)
                lp = dist.log_prob(act_arr[mb])
                ent = dist.entropy()

                ratio = torch.exp(lp - old_lp_t[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - self.clip_eps,
                                 1 + self.clip_eps) * adv_t[mb]
                loss = -torch.min(s1, s2).mean() - 0.01 * ent.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.gcn.parameters()), 0.5
                )
                self.optimizer.step()
                total_loss += loss.item()

        self._buffer.clear()
        return total_loss

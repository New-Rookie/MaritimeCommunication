"""
Improved MATD3 — Multi-Agent Twin Delayed DDPG with temporal encoder
and hierarchical experience replay for closed-loop MEC resource management.

Enhancements over standard MATD3 (Manuscript III §7):
  * Causal temporal encoder (2-layer Conv1d, kernel=3) over K_hist observations
  * Hierarchical replay: separate buffers for success / violation trajectories
  * Sigmoid-bounded scalar actions + Softmax storage splits
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from Env.phy import shannon_rate

from P3.resource_mgmt.task_offloader import (
    QueueState, simulate_offloading, select_source_buoys,
    find_local_candidates, find_edge_candidates,
)
from P3.resource_mgmt.metrics import aggregate_results, compute_reward

# ═══════════════════════════════════════════════════════════════════════════
# Dimensions
# ═══════════════════════════════════════════════════════════════════════════

OBS_DIM = 18
ACT_DIM = 6      # alpha_off, bw_frac, f_frac, omega_comm, omega_comp, omega_rem


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Encoder (causal Conv1d)
# ═══════════════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    """Two-layer causal 1-D convolution over K_hist time steps."""

    def __init__(self, in_dim: int, hidden: int = 64, kernel: int = 3):
        super().__init__()
        self.pad = kernel - 1
        self.conv1 = nn.Conv1d(in_dim, hidden, kernel_size=kernel)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, hidden)"""
        x = x.transpose(1, 2)                         # (B, C, T)
        x = F.pad(x, (self.pad, 0))
        x = F.relu(self.conv1(x))
        x = F.pad(x, (self.pad, 0))
        x = F.relu(self.conv2(x))
        return x[:, :, -1]                             # last time step


# ═══════════════════════════════════════════════════════════════════════════
# Actor / Critic networks
# ═══════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.net(obs)
        return torch.sigmoid(raw)


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, total_act_dim: int, hidden: int = 256):
        super().__init__()
        inp = state_dim + total_act_dim
        self.q1 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(inp, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor):
        sa = torch.cat([state, actions], dim=-1)
        return self.q1(sa).squeeze(-1), self.q2(sa).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchical Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalReplay:
    """Two-tier replay: success trajectories and violation trajectories."""

    def __init__(self, capacity: int = 50000):
        self.success_buf: deque = deque(maxlen=capacity)
        self.violate_buf: deque = deque(maxlen=capacity)

    def push(self, transition: Tuple, success: bool):
        if success:
            self.success_buf.append(transition)
        else:
            self.violate_buf.append(transition)

    def sample(self, batch_size: int, success_ratio: float = 0.6):
        n_suc = min(int(batch_size * success_ratio), len(self.success_buf))
        n_vio = min(batch_size - n_suc, len(self.violate_buf))
        n_suc = batch_size - n_vio

        items = []
        if n_suc > 0 and len(self.success_buf) > 0:
            idxs = np.random.randint(0, len(self.success_buf), size=n_suc)
            items.extend(self.success_buf[i] for i in idxs)
        if n_vio > 0 and len(self.violate_buf) > 0:
            idxs = np.random.randint(0, len(self.violate_buf), size=n_vio)
            items.extend(self.violate_buf[i] for i in idxs)
        if not items:
            return None
        return list(zip(*items))

    def __len__(self):
        return len(self.success_buf) + len(self.violate_buf)


# ═══════════════════════════════════════════════════════════════════════════
# Improved MATD3 Agent
# ═══════════════════════════════════════════════════════════════════════════

class ImprovedMATD3:
    """
    Multi-Agent TD3 with temporal encoder and hierarchical replay
    for closed-loop resource management.
    """

    def __init__(
        self,
        n_agents: int,
        cfg: EnvConfig,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_delay: int = 2,
        noise_std: float = 0.1,
        noise_clip: float = 0.3,
        batch_size: int = 256,
        K_hist: int = 6,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.cfg = cfg
        self.gamma = gamma
        self.tau = tau
        self.policy_delay = policy_delay
        self.noise_std = noise_std
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.K_hist = K_hist
        self.device = torch.device(device)
        self._update_count = 0

        enc_dim = 64
        self.encoder = TemporalEncoder(OBS_DIM, enc_dim, kernel=3).to(self.device)
        self.actor = Actor(enc_dim, ACT_DIM).to(self.device)

        global_state_dim = n_agents * enc_dim
        total_act_dim = n_agents * ACT_DIM
        self.critic = TwinCritic(global_state_dim, total_act_dim).to(self.device)

        self.encoder_target = copy.deepcopy(self.encoder)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.actor.parameters()), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.replay = HierarchicalReplay(capacity=50000)
        self._obs_history: Dict[int, deque] = {}

    # ─── observation construction ─────────────────────────────────────

    def build_observations(
        self, env: MarineIoTEnv, source_ids: List[int],
        local_map: Dict[int, int], edge_map: Dict[int, int],
    ) -> Dict[int, np.ndarray]:
        """Build per-agent observation vector (OBS_DIM,)."""
        cfg = self.cfg
        ceilings = cfg.scaled_ceilings()
        obs_dict: Dict[int, np.ndarray] = {}

        for bid in source_ids:
            o = np.zeros(OBS_DIM, dtype=np.float32)
            buoy = env.nodes[bid]
            o[0:3] = buoy.position / 100_000.0
            o[3:6] = buoy.velocity / 100.0
            o[6] = buoy.energy_residual / 100.0

            lid = local_map.get(bid, -1)
            if lid >= 0:
                lp = env.link_phy.get((bid, lid))
                o[7] = min(lp.sinr, 100.0) / 100.0 if lp else 0.0
                o[8] = min(lp.distance, cfg.R_local_buoy) / cfg.R_local_buoy if lp else 1.0
                lt = env.nodes[lid].node_type
                cl = ceilings.get(lt, ceilings["ship"])
                o[9] = cl["B_max"] / 50e6
                o[10] = cl["F_max"] / 32e9
                o[11] = cl["S_max"] / 50e9
            else:
                o[7:12] = 0.0

            eid = edge_map.get(bid, -1)
            if eid >= 0:
                et = env.nodes[eid].node_type
                ce = ceilings.get(et, ceilings["ship"])
                o[12] = ce["F_max"] / 32e9
            else:
                o[12] = 0.0

            o[13] = cfg.M_b / cfg.M_tot
            o[14] = 1.0
            o[15] = 1.0
            o[16] = float(env._step_count) / max(env.max_steps, 1)
            o[17] = 0.0
            obs_dict[bid] = o

        return obs_dict

    def _get_history(self, bid: int, obs: np.ndarray) -> np.ndarray:
        """Maintain and return K_hist observation history for agent."""
        if bid not in self._obs_history:
            self._obs_history[bid] = deque(maxlen=self.K_hist)
            for _ in range(self.K_hist):
                self._obs_history[bid].append(np.zeros(OBS_DIM, dtype=np.float32))
        self._obs_history[bid].append(obs)
        return np.stack(list(self._obs_history[bid]))   # (K_hist, OBS_DIM)

    # ─── action selection ─────────────────────────────────────────────

    @torch.no_grad()
    def select_actions(
        self, env: MarineIoTEnv, source_ids: List[int],
        local_map: Dict[int, int], edge_map: Dict[int, int],
        explore: bool = True,
    ) -> Tuple[Dict[int, Dict[str, float]], np.ndarray]:
        """Select resource-allocation actions for all source buoys."""
        obs_dict = self.build_observations(env, source_ids, local_map, edge_map)
        actions_out: Dict[int, Dict[str, float]] = {}
        encoded_list = []

        for bid in source_ids:
            hist = self._get_history(bid, obs_dict[bid])
            h_t = torch.FloatTensor(hist).unsqueeze(0).to(self.device)
            enc = self.encoder(h_t).squeeze(0)
            encoded_list.append(enc.cpu().numpy())

            a_t = self.actor(enc.unsqueeze(0)).squeeze(0).cpu().numpy()

            if explore:
                noise = np.random.randn(ACT_DIM).astype(np.float32) * self.noise_std
                a_t = np.clip(a_t + noise, 0.0, 1.0)

            actions_out[bid] = {
                "local_id": local_map.get(bid, -1),
                "edge_id": edge_map.get(bid, -1),
                "alpha_off": float(a_t[0]),
                "bw_frac": float(a_t[1]),
                "f_frac": float(a_t[2]),
                "omega_comm": float(a_t[3]),
                "omega_comp": float(a_t[4]),
                "omega_rem": float(a_t[5]),
            }

        return actions_out, np.stack(encoded_list) if encoded_list else np.zeros((0, 64))

    # ─── MEC assignment helpers ───────────────────────────────────────

    def build_assignments(
        self, env: MarineIoTEnv, source_ids: List[int],
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Assign local and edge MEC nodes to each source buoy."""
        cfg = self.cfg
        local_map: Dict[int, int] = {}
        edge_map: Dict[int, int] = {}

        for bid in source_ids:
            cands = find_local_candidates(env, bid, cfg)
            if cands:
                local_map[bid] = cands[0][0]
                e_cands = find_edge_candidates(env, cands[0][0], cfg)
                if e_cands:
                    edge_map[bid] = e_cands[0][0]
                else:
                    edge_map[bid] = -1
            else:
                local_map[bid] = -1
                edge_map[bid] = -1

        return local_map, edge_map

    # ─── training episode ─────────────────────────────────────────────

    def train_episode(
        self, env: MarineIoTEnv, n_windows: int = 5,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.cfg
        obs, info = env.reset()
        source_ids = select_source_buoys(env.nodes, cfg.N_src, rng)
        self._obs_history.clear()
        queue = QueueState()

        ep_rewards, ep_T, ep_E, ep_G = [], [], [], []

        for w in range(n_windows):
            env.recompute_ground_truth()
            local_map, edge_map = self.build_assignments(env, source_ids)

            obs_pre = self.build_observations(env, source_ids, local_map, edge_map)
            actions, enc_pre = self.select_actions(
                env, source_ids, local_map, edge_map, explore=True)

            results = simulate_offloading(env, cfg, source_ids, actions, queue)
            metrics = aggregate_results(results, cfg.Gamma_max)
            reward = compute_reward(metrics, cfg.T_max, cfg.E_max, cfg.Gamma_max)

            ep_rewards.append(reward)
            ep_T.append(metrics["mean_T_total"])
            ep_E.append(metrics["mean_E_total"])
            ep_G.append(metrics["mean_Gamma"])

            actions_env = np.ones((len(env.nodes), 2), dtype=np.float32)
            obs_new, _, term, trunc, _ = env.step(actions_env)

            obs_post = self.build_observations(env, source_ids, local_map, edge_map)

            global_s = np.concatenate([obs_pre[b] for b in source_ids])
            global_s_next = np.concatenate([obs_post[b] for b in source_ids])
            flat_a = np.concatenate([
                np.array([actions[b]["alpha_off"], actions[b]["bw_frac"],
                          actions[b]["f_frac"], actions[b].get("omega_comm", 0.5),
                          actions[b].get("omega_comp", 0.3),
                          actions[b].get("omega_rem", 0.2)])
                for b in source_ids
            ])
            done = 1.0 if (term or trunc) else 0.0
            avg_success = metrics["success_rate"] > 0.5

            self.replay.push(
                (global_s, flat_a, reward, global_s_next, done),
                success=avg_success,
            )

            if len(self.replay) >= self.batch_size:
                self._update()

            if term or trunc:
                obs_new, _ = env.reset()

        p_loss = getattr(self, "_last_p_loss", 0.0)
        v_loss = getattr(self, "_last_v_loss", 0.0)

        return {
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_T_total": float(np.mean(ep_T)),
            "mean_E_total": float(np.mean(ep_E)),
            "mean_Gamma": float(np.mean(ep_G)),
            "policy_loss": p_loss,
            "value_loss": v_loss,
        }

    # ─── evaluation window ────────────────────────────────────────────

    def eval_window(
        self, env: MarineIoTEnv,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, float]:
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.cfg
        source_ids = select_source_buoys(env.nodes, cfg.N_src, rng)
        env.recompute_ground_truth()
        local_map, edge_map = self.build_assignments(env, source_ids)
        actions, _ = self.select_actions(
            env, source_ids, local_map, edge_map, explore=False)
        queue = QueueState()
        results = simulate_offloading(env, cfg, source_ids, actions, queue)
        metrics = aggregate_results(results, cfg.Gamma_max)

        actions_env = np.ones((len(env.nodes), 2), dtype=np.float32)
        env.step(actions_env)

        return metrics

    # ─── TD3 update ───────────────────────────────────────────────────

    def _update(self):
        batch = self.replay.sample(self.batch_size)
        if batch is None:
            return

        states, actions, rewards, next_states, dones = batch
        s = torch.FloatTensor(np.stack(states)).to(self.device)
        a = torch.FloatTensor(np.stack(actions)).to(self.device)
        r = torch.FloatTensor(np.array(rewards, dtype=np.float32)).to(self.device)
        s2 = torch.FloatTensor(np.stack(next_states)).to(self.device)
        d = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.noise_std).clamp(
                -self.noise_clip, self.noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(0.0, 1.0)
            q1_tgt, q2_tgt = self.critic_target(s2, a2)
            q_target = r + self.gamma * (1 - d) * torch.min(q1_tgt, q2_tgt)

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()
        self._last_v_loss = critic_loss.item()

        self._update_count += 1
        if self._update_count % self.policy_delay == 0:
            actor_actions = self.actor(s)
            q1_val, _ = self.critic(s, actor_actions)
            actor_loss = -q1_val.mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) +
                list(self.actor.parameters()), 1.0)
            self.actor_optim.step()
            self._last_p_loss = actor_loss.item()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.encoder, self.encoder_target)
            self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

"""
GMAPPO — Graph-attention Multi-Agent PPO for MEC-aware link selection.

Architecture (Manuscript II Section 6):
  * 2-layer GCN encoder aggregating neighbor node & edge features
  * Per-agent actor: GCN-encoded obs -> discrete action (next-hop selection)
  * Global critic: concatenated graph summary -> V(s)
  * Action masking: only feasible next-hops selectable
  * Reward: mean_b(LA_pi_b) - eta_sw * N_switch - penalties
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from Env.channel import link_class as get_link_class

from P2.link_quality.metrics import (
    compute_let, compute_p_surv, compute_s_ho,
    path_quality, path_stability, link_advantage, compute_lqi,
)
from P2.link_quality.path_manager import PathManager, ServicePath
from P2.link_quality.rf_estimator import LinkQualityEstimator


# ═══════════════════════════════════════════════════════════════════════════
# GCN Encoder
# ═══════════════════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """Single graph convolution layer with edge features."""

    def __init__(self, node_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.W_node = nn.Linear(node_dim, out_dim)
        self.W_edge = nn.Linear(edge_dim, out_dim)
        self.W_self = nn.Linear(node_dim, out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        """
        h:         (B, N, node_dim)
        adj:       (B, N, N)  binary adjacency
        edge_feat: (B, N, N, edge_dim)
        returns:   (B, N, out_dim)
        """
        nbr_msg = self.W_node(h).unsqueeze(1).expand_as(
            edge_feat[:, :, :, :self.W_node.out_features].
            new_zeros(h.shape[0], h.shape[1], h.shape[1],
                      self.W_node.out_features))
        nbr_msg = self.W_node(h).unsqueeze(2).expand(
            h.shape[0], h.shape[1], h.shape[1], self.W_node.out_features)
        edge_msg = self.W_edge(edge_feat)
        combined = (nbr_msg + edge_msg) * adj.unsqueeze(-1)
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        agg = combined.sum(dim=2) / degree
        out = F.relu(agg + self.W_self(h))
        return out


class GCNEncoder(nn.Module):
    """Two-layer GCN from Manuscript II."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int = 64):
        super().__init__()
        self.layer1 = GCNLayer(node_dim, edge_dim, hidden)
        self.layer2 = GCNLayer(hidden, edge_dim, hidden)

    def forward(self, h: torch.Tensor, adj: torch.Tensor,
                edge_feat: torch.Tensor) -> torch.Tensor:
        h1 = self.layer1(h, adj, edge_feat)
        h2 = self.layer2(h1, adj, edge_feat)
        return h2


# ═══════════════════════════════════════════════════════════════════════════
# Actor / Critic
# ═══════════════════════════════════════════════════════════════════════════

class GCNActor(nn.Module):
    """GCN-encoded observation -> discrete action logits with masking."""

    def __init__(self, gcn_dim: int, max_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gcn_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, max_actions),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Categorical:
        logits = self.net(h)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits)


class GlobalCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Rollout buffer
# ═══════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self):
        self.obs, self.global_states, self.actions = [], [], []
        self.log_probs, self.rewards, self.dones, self.values = [], [], [], []
        self.masks = []

    def store(self, obs, gs, actions, lp, reward, done, value, masks):
        self.obs.append(obs)
        self.global_states.append(gs)
        self.actions.append(actions)
        self.log_probs.append(lp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.masks.append(masks)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ═══════════════════════════════════════════════════════════════════════════
# GMAPPO Agent
# ═══════════════════════════════════════════════════════════════════════════

NODE_FEAT_DIM = 10
EDGE_FEAT_DIM = 5
MAX_ACTIONS = 16   # K_nbr + stay + margin

class GMAPPO:
    """Graph-attention Multi-Agent PPO for service-path selection."""

    def __init__(self, n_agents: int, cfg: EnvConfig,
                 estimator: LinkQualityEstimator,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, entropy_coeff: float = 0.01,
                 n_epochs: int = 4, batch_size: int = 64,
                 device: str = "cpu"):
        self.n_agents = n_agents
        self.cfg = cfg
        self.estimator = estimator
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        gcn_hidden = 64
        self.gcn = GCNEncoder(NODE_FEAT_DIM, EDGE_FEAT_DIM,
                              gcn_hidden).to(self.device)
        self.actor = GCNActor(gcn_hidden, MAX_ACTIONS).to(self.device)

        global_state_dim = n_agents * gcn_hidden + 4
        self.critic = GlobalCritic(global_state_dim).to(self.device)

        params = (list(self.gcn.parameters()) +
                  list(self.actor.parameters()))
        self.actor_optim = torch.optim.Adam(params, lr=lr)
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer()
        self.path_mgr = PathManager(cfg)
        self._prev_actions: Dict[int, int] = {}
        self._sinr_histories: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    # ─── feature construction ─────────────────────────────────────────

    def build_node_features(self, env: MarineIoTEnv) -> np.ndarray:
        """(N, NODE_FEAT_DIM) node feature matrix."""
        nodes = env.nodes
        n = len(nodes)
        feats = np.zeros((n, NODE_FEAT_DIM), dtype=np.float32)
        for i, nd in enumerate(nodes):
            type_enc = {"satellite": 0, "uav": 1, "ship": 2,
                        "buoy": 3, "land": 4}
            feats[i, 0] = type_enc.get(nd.node_type, -1)
            feats[i, 1:4] = nd.position / 100_000.0
            feats[i, 4:7] = nd.velocity / 100.0
            feats[i, 7] = nd.energy_residual / 100.0
            feats[i, 8] = nd.tx_power * 1e3
            sinr_vals = [lp.sinr for (tx, rx), lp in env.link_phy.items()
                         if rx == nd.node_id]
            feats[i, 9] = float(np.mean(sinr_vals)) if sinr_vals else 0.0
        return feats

    def build_adj_and_edge_features(
            self, env: MarineIoTEnv) -> Tuple[np.ndarray, np.ndarray]:
        """(N,N) adjacency and (N,N,EDGE_FEAT_DIM) edge features."""
        n = len(env.nodes)
        adj = np.zeros((n, n), dtype=np.float32)
        edge_f = np.zeros((n, n, EDGE_FEAT_DIM), dtype=np.float32)
        gamma_lin = self.cfg.gamma_link_linear

        for (tx_id, rx_id), lp in env.link_phy.items():
            if lp.snr < gamma_lin:
                continue
            adj[rx_id, tx_id] = 1.0
            lc = get_link_class(env.nodes[tx_id].node_type,
                                env.nodes[rx_id].node_type)
            q = self._predict_q(lp, lc, tx_id, rx_id)
            s = self._compute_s_ho_for_link(env, tx_id, rx_id)
            edge_f[rx_id, tx_id, 0] = q
            edge_f[rx_id, tx_id, 1] = s
            edge_f[rx_id, tx_id, 2] = min(lp.distance / 100_000.0, 1.0)
            edge_f[rx_id, tx_id, 3] = lp.doppler / 1000.0
            edge_f[rx_id, tx_id, 4] = {"satellite": 0, "uav_terrestrial": 1,
                                        "sea_surface": 2, "terrestrial": 3
                                        }.get(lc, 0) / 3.0
        return adj, edge_f

    def build_global_state(self, gcn_out: np.ndarray,
                           mean_la: float, n_switch: int) -> np.ndarray:
        flat = gcn_out.flatten()
        extras = np.array([mean_la, float(n_switch),
                           float(self.n_agents), 0.0], dtype=np.float32)
        return np.concatenate([flat, extras])

    # ─── action selection ─────────────────────────────────────────────

    @torch.no_grad()
    def select_actions(
            self, env: MarineIoTEnv, source_ids: List[int],
            candidate_map: Dict[int, List[int]]
    ) -> Tuple[Dict[int, int], np.ndarray, np.ndarray, np.ndarray]:
        """Select next-hop for each source buoy.

        Returns (action_dict, log_probs_arr, gcn_out_np, mask_np).
        """
        node_feats = self.build_node_features(env)
        adj, edge_f = self.build_adj_and_edge_features(env)

        h = torch.FloatTensor(node_feats).unsqueeze(0).to(self.device)
        a = torch.FloatTensor(adj).unsqueeze(0).to(self.device)
        ef = torch.FloatTensor(edge_f).unsqueeze(0).to(self.device)
        gcn_out = self.gcn(h, a, ef).squeeze(0)
        gcn_np = gcn_out.cpu().numpy()

        action_dict: Dict[int, int] = {}
        all_lp = np.zeros(len(source_ids), dtype=np.float32)
        all_masks = np.zeros((len(source_ids), MAX_ACTIONS), dtype=np.float32)

        for idx, bid in enumerate(source_ids):
            candidates = candidate_map.get(bid, [])
            mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
            mask[0] = 1.0  # stay
            for ci, cid in enumerate(candidates[:MAX_ACTIONS - 1]):
                mask[ci + 1] = 1.0

            h_agent = gcn_out[bid].unsqueeze(0)
            m_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            dist = self.actor(h_agent, m_t)
            action = dist.sample()
            lp = dist.log_prob(action)
            a_idx = action.item()

            if a_idx == 0:
                chosen = self._prev_actions.get(bid, candidates[0]
                                                 if candidates else bid)
            elif a_idx - 1 < len(candidates):
                chosen = candidates[a_idx - 1]
            else:
                chosen = self._prev_actions.get(bid, bid)

            action_dict[bid] = chosen
            all_lp[idx] = lp.item()
            all_masks[idx] = mask

        return action_dict, all_lp, gcn_np, all_masks

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        return self.critic(gs_t).item()

    # ─── environment evaluation window ────────────────────────────────

    def run_window(self, env: MarineIoTEnv, source_ids: List[int],
                   n_steps: int = 20,
                   rng: Optional[np.random.Generator] = None
                   ) -> Dict[str, float]:
        """Run one evaluation window, compute LA metrics."""
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.cfg
        nodes = env.nodes
        n = len(nodes)

        candidate_map = self._build_candidate_map(env, source_ids)
        action_dict, _, gcn_np, masks = self.select_actions(
            env, source_ids, candidate_map)

        n_switch = self._count_switches(action_dict)
        self._prev_actions.update(action_dict)

        path_las = []
        for bid in source_ids:
            chosen_next = action_dict.get(bid, bid)
            hop_q, hop_s = self._evaluate_hops_for_buoy(
                env, bid, chosen_next)
            if hop_q:
                q_pi = path_quality(hop_q)
                s_pi = path_stability(hop_s)
                la = link_advantage(q_pi, s_pi, cfg.w_Q, cfg.w_S)
                path_las.append(la)

        mean_la = float(np.mean(path_las)) if path_las else 0.0
        n_outage = sum(1 for bid in source_ids
                       if bid not in action_dict or
                       action_dict[bid] == bid)

        actions_env = np.ones((n, 2), dtype=np.float32)
        for _ in range(n_steps):
            obs, _, term, trunc, _ = env.step(actions_env)
            if term or trunc:
                break
            self._update_sinr_histories(env)

        return {
            "mean_LA": mean_la,
            "n_switch": n_switch,
            "n_outage": n_outage,
            "path_las": path_las,
        }

    # ─── training ─────────────────────────────────────────────────────

    def train_episode(self, env: MarineIoTEnv, n_windows: int = 10,
                      rng: Optional[np.random.Generator] = None) -> Dict:
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.cfg
        obs, info = env.reset()
        nodes = env.nodes
        n = len(nodes)
        source_ids = PathManager.select_source_buoys(nodes, cfg.N_src, rng)
        self._prev_actions.clear()
        self._sinr_histories.clear()

        ep_rewards, ep_la = [], []
        collected_lp_tensors = []
        collected_rewards = []
        collected_gs = []
        collected_values = []

        for w in range(n_windows):
            env.recompute_ground_truth()
            self._update_sinr_histories(env)

            candidate_map = self._build_candidate_map(env, source_ids)

            # Forward pass WITH gradients for actor
            node_feats = self.build_node_features(env)
            adj, edge_f = self.build_adj_and_edge_features(env)
            h = torch.FloatTensor(node_feats).unsqueeze(0).to(self.device)
            a = torch.FloatTensor(adj).unsqueeze(0).to(self.device)
            ef = torch.FloatTensor(edge_f).unsqueeze(0).to(self.device)
            gcn_out = self.gcn(h, a, ef).squeeze(0)

            action_dict = {}
            window_lp = []
            for idx, bid in enumerate(source_ids):
                candidates = candidate_map.get(bid, [])
                mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
                mask[0] = 1.0
                for ci in range(min(len(candidates), MAX_ACTIONS - 1)):
                    mask[ci + 1] = 1.0
                h_agent = gcn_out[bid].unsqueeze(0)
                m_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                dist = self.actor(h_agent, m_t)
                action = dist.sample()
                lp = dist.log_prob(action)
                window_lp.append(lp)
                a_idx = action.item()
                if a_idx == 0:
                    chosen = self._prev_actions.get(bid,
                             candidates[0] if candidates else bid)
                elif a_idx - 1 < len(candidates):
                    chosen = candidates[a_idx - 1]
                else:
                    chosen = self._prev_actions.get(bid, bid)
                action_dict[bid] = chosen

            n_switch = self._count_switches(action_dict)
            self._prev_actions.update(action_dict)

            path_las = []
            for bid in source_ids:
                chosen_next = action_dict.get(bid, bid)
                hop_q, hop_s = self._evaluate_hops_for_buoy(
                    env, bid, chosen_next)
                if hop_q:
                    q_pi = path_quality(hop_q)
                    s_pi = path_stability(hop_s)
                    la = link_advantage(q_pi, s_pi, cfg.w_Q, cfg.w_S)
                    path_las.append(la)

            mean_la = float(np.mean(path_las)) if path_las else 0.0
            n_outage = sum(1 for bid in source_ids
                           if not candidate_map.get(bid))
            reward = mean_la - cfg.eta_sw * n_switch - 0.1 * n_outage
            ep_rewards.append(reward)
            ep_la.append(mean_la)

            if window_lp:
                collected_lp_tensors.append(torch.stack(window_lp))
            collected_rewards.append(reward)

            gcn_np = gcn_out.detach().cpu().numpy()
            gs = self.build_global_state(gcn_np, mean_la, n_switch)
            collected_gs.append(gs)
            collected_values.append(self.get_value(gs))

            actions_env = np.ones((n, 2), dtype=np.float32)
            obs, _, term, trunc, _ = env.step(actions_env)
            if term or trunc:
                obs, _ = env.reset()
                nodes = env.nodes

        # Policy gradient update
        p_loss_val = 0.0
        if collected_lp_tensors:
            rewards_t = torch.FloatTensor(collected_rewards).to(self.device)
            baseline = rewards_t.mean()
            advantages = rewards_t - baseline
            policy_loss = torch.tensor(0.0, device=self.device)
            for t, lp_t in enumerate(collected_lp_tensors):
                policy_loss += -(lp_t * advantages[t]).mean()
            policy_loss /= len(collected_lp_tensors)

            self.actor_optim.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.gcn.parameters()) +
                list(self.actor.parameters()), 0.5)
            self.actor_optim.step()
            p_loss_val = policy_loss.item()

        # Critic update
        v_loss_val = 0.0
        if collected_gs:
            gs_t = torch.FloatTensor(np.stack(collected_gs)).to(self.device)
            ret_t = torch.FloatTensor(collected_rewards).to(self.device)
            v_pred = self.critic(gs_t)
            v_loss = F.mse_loss(v_pred, ret_t)
            self.critic_optim.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()
            v_loss_val = v_loss.item()

        self.buffer.clear()

        return {
            "mean_reward": float(np.mean(ep_rewards)),
            "mean_LA": float(np.mean(ep_la)),
            "policy_loss": p_loss_val,
            "value_loss": v_loss_val,
        }

    # ─── PPO update ───────────────────────────────────────────────────

    def _ppo_update(self, n_src: int) -> Dict[str, float]:
        buf = self.buffer
        if len(buf) < 2:
            buf.clear()
            return {"policy_loss": 0.0, "value_loss": 0.0}

        T = len(buf.rewards)
        rewards = np.stack(buf.rewards)
        values = np.stack(buf.values)
        dones = np.array(buf.dones, dtype=np.float32)
        log_probs = np.stack(buf.log_probs)
        gs_arr = np.stack(buf.global_states)

        advantages = np.zeros_like(rewards)
        gae = np.zeros(n_src, dtype=np.float32)
        for t in reversed(range(T)):
            next_val = values[t + 1] if t + 1 < T else values[-1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        flat_lp = log_probs.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_gs = np.repeat(gs_arr, n_src, axis=0)

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        lp_t = torch.FloatTensor(flat_lp).to(self.device)
        lp_t.requires_grad_(False)
        adv_t = torch.FloatTensor(flat_adv).to(self.device)
        ret_t = torch.FloatTensor(flat_ret).to(self.device)
        gs_t = torch.FloatTensor(flat_gs).to(self.device)

        p_losses, v_losses = [], []

        for _ in range(self.n_epochs):
            # Policy gradient: -log_prob * advantage (REINFORCE with baseline)
            p_loss = -(lp_t * adv_t).mean()
            # This is detached from actor params, so we use the critic update
            # as the main learning signal and treat actor as frozen per-buffer

            v_pred = self.critic(gs_t)
            v_loss = F.mse_loss(v_pred, ret_t)

            self.critic_optim.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()

            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())

        buf.clear()
        return {
            "policy_loss": float(np.mean(p_losses)) if p_losses else 0.0,
            "value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
        }

    # ─── internal helpers ─────────────────────────────────────────────

    def _build_candidate_map(self, env: MarineIoTEnv,
                             source_ids: List[int]) -> Dict[int, List[int]]:
        """For each source buoy, list candidate first-hop node IDs."""
        cmap: Dict[int, List[int]] = {}
        gamma_lin = self.cfg.gamma_link_linear
        ships_uavs = [nd.node_id for nd in env.nodes
                      if nd.node_type in ("ship", "uav")]
        for bid in source_ids:
            scored = []
            for cid in ships_uavs:
                lp = env.link_phy.get((bid, cid))
                if lp is not None and lp.snr >= gamma_lin:
                    scored.append((cid, lp.p_sig))
            scored.sort(key=lambda x: x[1], reverse=True)
            cmap[bid] = [c for c, _ in scored[:self.cfg.K_nbr]]
        return cmap

    def _count_switches(self, action_dict: Dict[int, int]) -> int:
        count = 0
        for bid, chosen in action_dict.items():
            prev = self._prev_actions.get(bid)
            if prev is not None and prev != chosen:
                count += 1
        return count

    def _predict_q(self, lp, lc: str, tx_id: int, rx_id: int) -> float:
        if not self.estimator.is_trained:
            sinr = max(lp.sinr, 1e-30)
            import math
            ber = 0.5 * math.erfc(math.sqrt(sinr))
            return max(0.0, (1.0 - ber) ** self.cfg.L_pkt)

        key = (tx_id, rx_id)
        hist = self._sinr_histories.get(key, [lp.sinr])
        sinr_arr = np.array(hist[-10:])
        rssi_arr = np.array([lp.rssi] * len(sinr_arr))

        return self.estimator.predict_single(
            lc, float(lp.rssi), float(lp.snr), float(lp.sinr),
            compute_lqi(lp.sinr),
            float(np.mean(sinr_arr)), float(np.std(sinr_arr)),
            float(np.mean(rssi_arr)), float(np.std(rssi_arr)),
            float(lp.doppler), 0, len(hist))

    def _compute_s_ho_for_link(self, env: MarineIoTEnv,
                               tx_id: int, rx_id: int) -> float:
        tx_n = env.nodes[tx_id]
        rx_n = env.nodes[rx_id]
        dp = rx_n.position - tx_n.position
        dv = rx_n.velocity - tx_n.velocity
        from Env.phy import communication_range_estimate
        r_comm = communication_range_estimate(
            tx_n.node_type, rx_n.node_type, self.cfg)

        let = compute_let(dp, dv, r_comm)
        key = (tx_id, rx_id)
        hist = self._sinr_histories.get(key, [])
        if len(hist) < 2:
            lp = env.link_phy.get((tx_id, rx_id))
            sinr_val = lp.sinr if lp else 1.0
            hist = [sinr_val, sinr_val]
        sinr_arr = np.array(hist[-10:])
        p_surv = compute_p_surv(sinr_arr, self.cfg.gamma_ho_linear,
                                self.cfg.N_p, self.cfg.delta_t_sim * 1e-3)
        tau_req_s = self.cfg.tau_req * 1e-3
        return compute_s_ho(let, p_surv, tau_req_s)

    def _evaluate_hops_for_buoy(
            self, env: MarineIoTEnv, bid: int, first_hop: int
    ) -> Tuple[List[float], List[float]]:
        """Build a minimal path and return per-hop Q and S_HO lists."""
        nodes = env.nodes
        hop_q, hop_s = [], []

        lp = env.link_phy.get((bid, first_hop))
        if lp is None:
            return [], []
        lc = get_link_class(nodes[bid].node_type, nodes[first_hop].node_type)
        hop_q.append(self._predict_q(lp, lc, bid, first_hop))
        hop_s.append(self._compute_s_ho_for_link(env, bid, first_hop))

        sats = [nd for nd in nodes if nd.node_type == "satellite"]
        best_sat, best_sig = None, -1.0
        for sat in sats:
            lp_s = env.link_phy.get((first_hop, sat.node_id))
            if lp_s and lp_s.snr >= self.cfg.gamma_link_linear:
                if lp_s.p_sig > best_sig:
                    best_sat = sat
                    best_sig = lp_s.p_sig
        if best_sat is None:
            return [], []
        lp_s = env.link_phy[(first_hop, best_sat.node_id)]
        lc_s = get_link_class(nodes[first_hop].node_type, best_sat.node_type)
        hop_q.append(self._predict_q(lp_s, lc_s, first_hop, best_sat.node_id))
        hop_s.append(self._compute_s_ho_for_link(env, first_hop, best_sat.node_id))

        lands = [nd for nd in nodes if nd.node_type == "land"]
        best_land, best_sig_l = None, -1.0
        for ld in lands:
            lp_l = env.link_phy.get((best_sat.node_id, ld.node_id))
            if lp_l and lp_l.p_sig > best_sig_l:
                best_land = ld
                best_sig_l = lp_l.p_sig
        if best_land is None:
            return [], []
        lp_l = env.link_phy[(best_sat.node_id, best_land.node_id)]
        lc_l = get_link_class(best_sat.node_type, best_land.node_type)
        hop_q.append(self._predict_q(lp_l, lc_l, best_sat.node_id, best_land.node_id))
        hop_s.append(self._compute_s_ho_for_link(env, best_sat.node_id, best_land.node_id))

        return hop_q, hop_s

    def _update_sinr_histories(self, env: MarineIoTEnv):
        for (tx_id, rx_id), lp in env.link_phy.items():
            key = (tx_id, rx_id)
            self._sinr_histories[key].append(lp.sinr)
            if len(self._sinr_histories[key]) > 20:
                self._sinr_histories[key] = self._sinr_histories[key][-20:]

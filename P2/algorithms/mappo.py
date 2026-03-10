"""
MAPPO — Multi-Agent PPO baseline for link selection (no GCN).

Replaces the GCN encoder with a standard MLP that concatenates local
observation with mean-pooled neighbor features.  Same global critic
and PPO update as GMAPPO.  Isolates the contribution of graph structure.
"""

from __future__ import annotations

import math
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
from P2.link_quality.path_manager import PathManager
from P2.link_quality.rf_estimator import LinkQualityEstimator
from P2.algorithms.gmappo import MAX_ACTIONS, NODE_FEAT_DIM


class MLPEncoder(nn.Module):
    """Local obs + mean-pooled neighbor features -> agent embedding."""

    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )

    def forward(self, local_obs: torch.Tensor,
                nbr_mean: torch.Tensor) -> torch.Tensor:
        x = torch.cat([local_obs, nbr_mean], dim=-1)
        return self.net(x)


class MAPPOActor(nn.Module):
    def __init__(self, embed_dim: int, max_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, max_actions),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> Categorical:
        logits = self.net(h)
        logits = logits.masked_fill(~mask.bool(), -1e9)
        return Categorical(logits=logits)


class MAPPOCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.obs, self.global_states = [], []
        self.log_probs, self.rewards, self.dones, self.values = [], [], [], []

    def store(self, obs, gs, lp, reward, done, value):
        self.obs.append(obs)
        self.global_states.append(gs)
        self.log_probs.append(lp)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


class MAPPO:
    """Standard MAPPO — MLP-based, no graph convolution."""

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

        embed_dim = 64
        self.encoder = MLPEncoder(NODE_FEAT_DIM, embed_dim).to(self.device)
        self.actor = MAPPOActor(embed_dim, MAX_ACTIONS).to(self.device)
        gs_dim = n_agents * embed_dim + 4
        self.critic = MAPPOCritic(gs_dim).to(self.device)

        params = list(self.encoder.parameters()) + list(self.actor.parameters())
        self.actor_optim = torch.optim.Adam(params, lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer()
        self.path_mgr = PathManager(cfg)
        self._prev_actions: Dict[int, int] = {}
        self._sinr_histories: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    def _build_node_features(self, env: MarineIoTEnv) -> np.ndarray:
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

    def _mean_pool_neighbors(self, feats: np.ndarray,
                             env: MarineIoTEnv) -> np.ndarray:
        n = len(env.nodes)
        pooled = np.zeros_like(feats)
        gamma_lin = self.cfg.gamma_link_linear
        for i in range(n):
            nbr_feats = []
            for (tx, rx), lp in env.link_phy.items():
                if rx == i and lp.snr >= gamma_lin:
                    nbr_feats.append(feats[tx])
            if nbr_feats:
                pooled[i] = np.mean(nbr_feats, axis=0)
        return pooled

    @torch.no_grad()
    def select_actions(self, env: MarineIoTEnv, source_ids: List[int],
                       candidate_map: Dict[int, List[int]]
                       ) -> Tuple[Dict[int, int], np.ndarray, np.ndarray]:
        feats = self._build_node_features(env)
        nbr_pool = self._mean_pool_neighbors(feats, env)
        feats_t = torch.FloatTensor(feats).to(self.device)
        nbr_t = torch.FloatTensor(nbr_pool).to(self.device)
        embeddings = self.encoder(feats_t, nbr_t)
        embed_np = embeddings.cpu().numpy()

        action_dict: Dict[int, int] = {}
        all_lp = np.zeros(len(source_ids), dtype=np.float32)

        for idx, bid in enumerate(source_ids):
            candidates = candidate_map.get(bid, [])
            mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
            mask[0] = 1.0
            for ci in range(min(len(candidates), MAX_ACTIONS - 1)):
                mask[ci + 1] = 1.0

            h = embeddings[bid].unsqueeze(0)
            m_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
            dist = self.actor(h, m_t)
            action = dist.sample()
            a_idx = action.item()

            if a_idx == 0:
                chosen = self._prev_actions.get(bid, candidates[0]
                                                 if candidates else bid)
            elif a_idx - 1 < len(candidates):
                chosen = candidates[a_idx - 1]
            else:
                chosen = self._prev_actions.get(bid, bid)

            action_dict[bid] = chosen
            all_lp[idx] = dist.log_prob(action).item()

        return action_dict, all_lp, embed_np

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

        for w in range(n_windows):
            env.recompute_ground_truth()
            self._update_sinr_histories(env)

            candidate_map = self._build_candidate_map(env, source_ids)

            # Forward with gradients
            feats = self._build_node_features(env)
            nbr_pool = self._mean_pool_neighbors(feats, env)
            feats_t = torch.FloatTensor(feats).to(self.device)
            nbr_t = torch.FloatTensor(nbr_pool).to(self.device)
            embeddings = self.encoder(feats_t, nbr_t)

            action_dict = {}
            window_lp = []
            for idx, bid in enumerate(source_ids):
                candidates = candidate_map.get(bid, [])
                mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
                mask[0] = 1.0
                for ci in range(min(len(candidates), MAX_ACTIONS - 1)):
                    mask[ci + 1] = 1.0
                h = embeddings[bid].unsqueeze(0)
                m_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                dist = self.actor(h, m_t)
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
                hop_q, hop_s = self._evaluate_hops(env, bid,
                                                   action_dict.get(bid, bid))
                if hop_q:
                    la = link_advantage(path_quality(hop_q),
                                        path_stability(hop_s),
                                        cfg.w_Q, cfg.w_S)
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

            embed_np = embeddings.detach().cpu().numpy()
            gs = np.concatenate([embed_np.flatten(),
                                 [mean_la, float(n_switch),
                                  float(n), 0.0]]).astype(np.float32)
            collected_gs.append(gs)

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
                list(self.encoder.parameters()) +
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
        return {"mean_reward": float(np.mean(ep_rewards)),
                "mean_LA": float(np.mean(ep_la)),
                "policy_loss": p_loss_val,
                "value_loss": v_loss_val}

    @torch.no_grad()
    def _get_value(self, gs: np.ndarray) -> float:
        gs_t = torch.FloatTensor(gs).unsqueeze(0).to(self.device)
        return self.critic(gs_t).item()

    def _ppo_update(self, n_src: int) -> Dict[str, float]:
        buf = self.buffer
        if len(buf) < 2:
            buf.clear()
            return {"policy_loss": 0.0, "value_loss": 0.0}

        T = len(buf.rewards)
        rewards = np.stack(buf.rewards)
        values = np.stack(buf.values)
        dones = np.array(buf.dones, dtype=np.float32)
        gs_arr = np.stack(buf.global_states)

        advantages = np.zeros_like(rewards)
        gae = np.zeros(n_src, dtype=np.float32)
        for t in reversed(range(T)):
            nv = values[t + 1] if t + 1 < T else values[-1]
            delta = rewards[t] + self.gamma * nv * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values

        flat_adv = (advantages.reshape(-1))
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        flat_ret = returns.reshape(-1)
        flat_gs = np.repeat(gs_arr, n_src, axis=0)

        adv_t = torch.FloatTensor(flat_adv).to(self.device)
        ret_t = torch.FloatTensor(flat_ret).to(self.device)
        gs_t = torch.FloatTensor(flat_gs).to(self.device)

        total = len(flat_adv)
        p_l, v_l = [], []
        for _ in range(self.n_epochs):
            idx = np.random.permutation(total)
            for s in range(0, total, self.batch_size):
                b = idx[s:s + self.batch_size]
                p_loss = -adv_t[b].mean()
                v_pred = self.critic(gs_t[b])
                v_loss = F.mse_loss(v_pred, ret_t[b])
                self.actor_optim.zero_grad()
                p_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) +
                    list(self.actor.parameters()), 0.5)
                self.actor_optim.step()
                self.critic_optim.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()
                p_l.append(p_loss.item())
                v_l.append(v_loss.item())
        buf.clear()
        return {"policy_loss": float(np.mean(p_l)) if p_l else 0.0,
                "value_loss": float(np.mean(v_l)) if v_l else 0.0}

    # ─── shared helpers (same logic as GMAPPO) ────────────────────────

    def _build_candidate_map(self, env, source_ids):
        cmap = {}
        gamma_lin = self.cfg.gamma_link_linear
        ships_uavs = [nd.node_id for nd in env.nodes
                      if nd.node_type in ("ship", "uav")]
        for bid in source_ids:
            scored = []
            for cid in ships_uavs:
                lp = env.link_phy.get((bid, cid))
                if lp and lp.snr >= gamma_lin:
                    scored.append((cid, lp.p_sig))
            scored.sort(key=lambda x: x[1], reverse=True)
            cmap[bid] = [c for c, _ in scored[:self.cfg.K_nbr]]
        return cmap

    def _count_switches(self, ad):
        return sum(1 for b, c in ad.items()
                   if self._prev_actions.get(b) is not None and
                   self._prev_actions[b] != c)

    def _predict_q(self, lp, lc, tx, rx):
        if not self.estimator.is_trained:
            sinr = max(lp.sinr, 1e-30)
            ber = 0.5 * math.erfc(math.sqrt(sinr))
            return max(0.0, (1.0 - ber) ** self.cfg.L_pkt)
        hist = self._sinr_histories.get((tx, rx), [lp.sinr])
        sa = np.array(hist[-10:])
        return self.estimator.predict_single(
            lc, float(lp.rssi), float(lp.snr), float(lp.sinr),
            compute_lqi(lp.sinr),
            float(np.mean(sa)), float(np.std(sa)),
            float(lp.rssi), 0.0,
            float(lp.doppler), 0, len(hist))

    def _compute_s_ho(self, env, tx_id, rx_id):
        from Env.phy import communication_range_estimate
        tx_n, rx_n = env.nodes[tx_id], env.nodes[rx_id]
        dp = rx_n.position - tx_n.position
        dv = rx_n.velocity - tx_n.velocity
        r = communication_range_estimate(tx_n.node_type, rx_n.node_type, self.cfg)
        let = compute_let(dp, dv, r)
        hist = self._sinr_histories.get((tx_id, rx_id), [])
        if len(hist) < 2:
            lp = env.link_phy.get((tx_id, rx_id))
            v = lp.sinr if lp else 1.0
            hist = [v, v]
        p = compute_p_surv(np.array(hist[-10:]), self.cfg.gamma_ho_linear,
                           self.cfg.N_p, self.cfg.delta_t_sim * 1e-3)
        return compute_s_ho(let, p, self.cfg.tau_req * 1e-3)

    def _evaluate_hops(self, env, bid, first_hop):
        nodes = env.nodes
        hq, hs = [], []
        lp = env.link_phy.get((bid, first_hop))
        if not lp:
            return [], []
        lc = get_link_class(nodes[bid].node_type, nodes[first_hop].node_type)
        hq.append(self._predict_q(lp, lc, bid, first_hop))
        hs.append(self._compute_s_ho(env, bid, first_hop))
        best_sat, best_sig = None, -1.0
        for nd in nodes:
            if nd.node_type != "satellite":
                continue
            lps = env.link_phy.get((first_hop, nd.node_id))
            if lps and lps.snr >= self.cfg.gamma_link_linear and lps.p_sig > best_sig:
                best_sat, best_sig = nd, lps.p_sig
        if not best_sat:
            return [], []
        lps = env.link_phy[(first_hop, best_sat.node_id)]
        lcs = get_link_class(nodes[first_hop].node_type, best_sat.node_type)
        hq.append(self._predict_q(lps, lcs, first_hop, best_sat.node_id))
        hs.append(self._compute_s_ho(env, first_hop, best_sat.node_id))
        best_land, best_sl = None, -1.0
        for nd in nodes:
            if nd.node_type != "land":
                continue
            lpl = env.link_phy.get((best_sat.node_id, nd.node_id))
            if lpl and lpl.p_sig > best_sl:
                best_land, best_sl = nd, lpl.p_sig
        if not best_land:
            return [], []
        lpl = env.link_phy[(best_sat.node_id, best_land.node_id)]
        lcl = get_link_class(best_sat.node_type, best_land.node_type)
        hq.append(self._predict_q(lpl, lcl, best_sat.node_id, best_land.node_id))
        hs.append(self._compute_s_ho(env, best_sat.node_id, best_land.node_id))
        return hq, hs

    def _update_sinr_histories(self, env):
        for (tx, rx), lp in env.link_phy.items():
            self._sinr_histories[(tx, rx)].append(lp.sinr)
            if len(self._sinr_histories[(tx, rx)]) > 20:
                self._sinr_histories[(tx, rx)] = self._sinr_histories[(tx, rx)][-20:]

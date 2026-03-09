"""
Greedy baseline for MEC-aware link selection.

For each source buoy, evaluates all feasible first-hop candidates and
selects the one leading to the highest immediate LA_pi.  No learning.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from Env.channel import link_class as get_link_class
from Env.phy import communication_range_estimate

from P2.link_quality.metrics import (
    compute_let, compute_p_surv, compute_s_ho,
    path_quality, path_stability, link_advantage, compute_lqi,
)
from P2.link_quality.path_manager import PathManager
from P2.link_quality.rf_estimator import LinkQualityEstimator


class GreedySelector:
    """Pick the first-hop that maximises immediate LA_pi."""

    def __init__(self, n_agents: int, cfg: EnvConfig,
                 estimator: LinkQualityEstimator):
        self.n_agents = n_agents
        self.cfg = cfg
        self.estimator = estimator
        self.path_mgr = PathManager(cfg)
        self._sinr_histories: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    def run_episode(self, env: MarineIoTEnv, n_windows: int = 10,
                    rng: Optional[np.random.Generator] = None) -> Dict:
        if rng is None:
            rng = np.random.default_rng()
        cfg = self.cfg
        obs, _ = env.reset()
        nodes = env.nodes
        n = len(nodes)
        source_ids = PathManager.select_source_buoys(nodes, cfg.N_src, rng)
        self._sinr_histories.clear()

        ep_la = []
        total_switches = 0
        prev_actions: Dict[int, int] = {}

        for w in range(n_windows):
            env.recompute_ground_truth()
            self._update_sinr(env)

            action_dict = self._select_greedy(env, source_ids)
            total_switches += sum(
                1 for b, c in action_dict.items()
                if prev_actions.get(b) is not None and prev_actions[b] != c)
            prev_actions.update(action_dict)

            path_las = []
            for bid in source_ids:
                hop_q, hop_s = self._evaluate_hops(
                    env, bid, action_dict.get(bid, bid))
                if hop_q:
                    la = link_advantage(path_quality(hop_q),
                                        path_stability(hop_s),
                                        cfg.w_Q, cfg.w_S)
                    path_las.append(la)
            ep_la.append(float(np.mean(path_las)) if path_las else 0.0)

            actions_env = np.ones((n, 2), dtype=np.float32)
            obs, _, term, trunc, _ = env.step(actions_env)
            if term or trunc:
                obs, _ = env.reset()
                nodes = env.nodes

        return {"mean_LA": float(np.mean(ep_la)),
                "n_switch": total_switches}

    def _select_greedy(self, env: MarineIoTEnv,
                       source_ids: List[int]) -> Dict[int, int]:
        gamma_lin = self.cfg.gamma_link_linear
        ships_uavs = [nd.node_id for nd in env.nodes
                      if nd.node_type in ("ship", "uav")]
        result: Dict[int, int] = {}
        for bid in source_ids:
            best_la, best_hop = -1.0, None
            for cid in ships_uavs:
                lp = env.link_phy.get((bid, cid))
                if lp is None or lp.snr < gamma_lin:
                    continue
                hop_q, hop_s = self._evaluate_hops(env, bid, cid)
                if hop_q:
                    la = link_advantage(path_quality(hop_q),
                                        path_stability(hop_s),
                                        self.cfg.w_Q, self.cfg.w_S)
                    if la > best_la:
                        best_la = la
                        best_hop = cid
            if best_hop is not None:
                result[bid] = best_hop
        return result

    # ─── hop evaluation (shared pattern) ──────────────────────────────

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

    def _update_sinr(self, env):
        for (tx, rx), lp in env.link_phy.items():
            self._sinr_histories[(tx, rx)].append(lp.sinr)
            if len(self._sinr_histories[(tx, rx)]) > 20:
                self._sinr_histories[(tx, rx)] = self._sinr_histories[(tx, rx)][-20:]

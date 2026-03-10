"""
Improved IPPO — Independent PPO with a Global Critic and
counterfactual difference rewards for INDP neighbour discovery.

Key features (from Manuscript I Section 8):
  * Per-agent actor MLP
  * Centralised global-state critic MLP (CTDE)
  * Counterfactual difference reward D_i^t = F1(a_all) - F1(a_{-i}, silent)
  * Active-node filtering + minibatch B_cf for scalability
  * Clipped PPO surrogate objective + GAE advantage estimation
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from Env.config import EnvConfig


# ═══════════════════════════════════════════════════════════════════════════
# Networks
# ═══════════════════════════════════════════════════════════════════════════

class ActorNetwork(nn.Module):
    """Per-agent actor: maps local observation to (mu, sigma) for actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        mu = torch.sigmoid(self.mu_head(h))  # bounded [0,1]
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_dist(self, obs: torch.Tensor) -> Normal:
        mu, std = self(obs)
        return Normal(mu, std + 1e-6)


class GlobalCritic(nn.Module):
    """Centralised critic: maps concatenated global state to V(s)."""

    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Rollout buffer
# ═══════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.global_states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[np.ndarray] = []
        self.rewards: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.values: List[np.ndarray] = []

    def store(self, obs, global_state, actions, log_probs, rewards, done, values):
        self.obs.append(obs)
        self.global_states.append(global_state)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.dones.append(done)
        self.values.append(values)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.obs)


# ═══════════════════════════════════════════════════════════════════════════
# Improved IPPO Agent
# ═══════════════════════════════════════════════════════════════════════════

class ImprovedIPPO:
    """
    Improved IPPO with Global Critic for INDP optimisation.

    O_i = [Type_i, p_i, v_i, E_res_i, Y_i, Nhat_env_i, c_coll(t-1), |LNT(t-1)|, m_i]
    S   = [{O_i}, A_gt, I_global, E_res_all]
    a_i = [tau_i, P_tx_i]   (continuous, bounded)
    r_i = w1*D_i - w2*E_ND_i/E_ref - w3*c_coll_i
    """

    def __init__(self, n_agents: int, obs_dim: int = 16, act_dim: int = 2,
                 lr: float = 3e-4, gamma: float = 0.99, lam: float = 0.95,
                 clip_eps: float = 0.2, entropy_coeff: float = 0.01,
                 n_epochs: int = 4, batch_size: int = 64,
                 cfg: Optional[EnvConfig] = None,
                 device: str = "cpu"):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coeff = entropy_coeff
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.cfg = cfg or EnvConfig()
        self.device = torch.device(device)

        # reward weights
        self.w1 = 1.0    # F1 difference
        self.w2 = 0.1    # energy penalty
        self.w3 = 0.05   # collision penalty

        # shared actor (parameter-shared across agents)
        self.actor = ActorNetwork(obs_dim, act_dim).to(self.device)
        # global critic
        global_state_dim = n_agents * obs_dim + n_agents + 2
        self.critic = GlobalCritic(global_state_dim).to(self.device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    # -----------------------------------------------------------------
    # Global state construction
    # -----------------------------------------------------------------

    def build_global_state(self, obs: np.ndarray,
                           gt_adj: Optional[np.ndarray] = None,
                           i_global: float = 0.0) -> np.ndarray:
        flat_obs = obs.flatten()
        e_res = obs[:, 7] if obs.ndim == 2 else np.zeros(self.n_agents)
        extras = np.array([i_global, float(np.mean(e_res))])
        return np.concatenate([flat_obs, e_res, extras]).astype(np.float32)

    # -----------------------------------------------------------------
    # Action selection
    # -----------------------------------------------------------------

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray,
                       global_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        obs_t = torch.FloatTensor(obs).to(self.device)
        gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)

        dist = self.actor.get_dist(obs_t)
        actions = dist.sample()
        actions = torch.clamp(actions, 0.0, 1.0)
        log_probs = dist.log_prob(actions).sum(-1)

        value = self.critic(gs_t)

        return (actions.cpu().numpy(),
                log_probs.cpu().numpy())

    @torch.no_grad()
    def get_value(self, global_state: np.ndarray) -> float:
        gs_t = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        return self.critic(gs_t).item()

    # -----------------------------------------------------------------
    # Counterfactual difference reward
    # -----------------------------------------------------------------

    def compute_counterfactual_rewards(
            self, f1_all: float, per_agent_f1_without: np.ndarray,
            energies: np.ndarray, collisions: np.ndarray) -> np.ndarray:
        """
        r_i = w1 * D_i - w2 * E_i / E_ref - w3 * c_i
        D_i = F1(a_all) - F1(a_{-i}, a_i=silent)
        """
        d_i = f1_all - per_agent_f1_without
        r = self.w1 * d_i - self.w2 * energies / self.cfg.E_ref - self.w3 * collisions
        return r.astype(np.float32)

    def compute_counterfactual_f1_batch(
            self, env, protocol, nodes, cfg, rng,
            active_ids: List[int], b_cf: int) -> np.ndarray:
        """
        Scalable counterfactual: evaluate only active transmitters,
        with minibatch B_cf.
        """
        n = len(nodes)
        f1_without = np.zeros(n, dtype=np.float32)

        # baseline F1 with all active
        gt = env.get_ground_truth_topology()
        f1_all, _, _, _ = protocol.compute_f1(gt, n)

        # subsample active set if too large
        eval_ids = active_ids
        if len(active_ids) > b_cf:
            eval_ids = list(rng.choice(active_ids, size=b_cf, replace=False))

        for nid in eval_ids:
            # temporarily silence node nid
            orig_power = nodes[nid].tx_power
            nodes[nid].tx_power = 0.0

            # re-run one window with this node silent
            silent_result = protocol.run_window(nodes, cfg, rng)
            f1_silent, _, _, _ = protocol.compute_f1(gt, n)
            f1_without[nid] = f1_silent

            nodes[nid].tx_power = orig_power

        # for non-evaluated agents, use the mean difference
        mean_diff = f1_all - np.mean(f1_without[eval_ids]) if eval_ids else 0.0
        for i in range(n):
            if i not in eval_ids:
                f1_without[i] = f1_all - mean_diff

        return f1_without

    # -----------------------------------------------------------------
    # GAE computation
    # -----------------------------------------------------------------

    def compute_gae(self, rewards: List[np.ndarray],
                    values: List[np.ndarray],
                    dones: List[bool],
                    last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        n_steps = len(rewards)
        advantages = np.zeros((n_steps, self.n_agents), dtype=np.float32)
        returns = np.zeros_like(advantages)
        gae = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_val = last_value
            else:
                next_val = values[t + 1]
            if isinstance(next_val, np.ndarray):
                next_val = np.mean(next_val)
            cur_val = values[t]
            if isinstance(cur_val, np.ndarray):
                cur_val = np.mean(cur_val)

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - cur_val
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + cur_val

        return advantages, returns

    # -----------------------------------------------------------------
    # PPO update
    # -----------------------------------------------------------------

    def update(self) -> Dict[str, float]:
        buf = self.buffer
        if len(buf) < 2:
            buf.clear()
            return {"policy_loss": 0, "value_loss": 0}

        last_val = buf.values[-1]
        if isinstance(last_val, np.ndarray):
            last_val = float(np.mean(last_val))

        advantages, returns = self.compute_gae(
            buf.rewards, buf.values, buf.dones, last_val)

        # flatten across time
        all_obs = np.stack(buf.obs)           # (T, N, obs_dim)
        all_gs = np.stack(buf.global_states)  # (T, gs_dim)
        all_act = np.stack(buf.actions)       # (T, N, act_dim)
        all_lp = np.stack(buf.log_probs)      # (T, N)

        T, N = all_obs.shape[:2]
        obs_flat = all_obs.reshape(T * N, -1)
        act_flat = all_act.reshape(T * N, -1)
        lp_flat = all_lp.reshape(T * N)
        adv_flat = advantages.reshape(T * N)
        ret_flat = returns.reshape(T * N)

        # repeat global states for each agent
        gs_flat = np.repeat(all_gs, N, axis=0)  # (T*N, gs_dim)

        # normalise advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # to tensors
        obs_t = torch.FloatTensor(obs_flat).to(self.device)
        act_t = torch.FloatTensor(act_flat).to(self.device)
        old_lp_t = torch.FloatTensor(lp_flat).to(self.device)
        adv_t = torch.FloatTensor(adv_flat).to(self.device)
        ret_t = torch.FloatTensor(ret_flat).to(self.device)
        gs_t = torch.FloatTensor(gs_flat).to(self.device)

        total_samples = T * N
        policy_losses = []
        value_losses = []

        for _ in range(self.n_epochs):
            indices = np.random.permutation(total_samples)
            for start in range(0, total_samples, self.batch_size):
                end = min(start + self.batch_size, total_samples)
                idx = indices[start:end]

                dist = self.actor.get_dist(obs_t[idx])
                new_lp = dist.log_prob(act_t[idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()

                ratio = (new_lp - old_lp_t[idx]).exp()
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                    1 + self.clip_eps) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy

                values = self.critic(gs_t[idx])
                value_loss = F.mse_loss(values, ret_t[idx])

                self.actor_optim.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())

        buf.clear()
        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
        }

    # -----------------------------------------------------------------
    # Training loop helper
    # -----------------------------------------------------------------

    def train_episode(self, env, protocol, n_windows: int = 10,
                      rng: Optional[np.random.Generator] = None) -> Dict:
        """Run one training episode of n_windows discovery windows."""
        if rng is None:
            rng = np.random.default_rng()

        cfg = self.cfg
        obs, info = env.reset()
        nodes = env.nodes
        n = len(nodes)

        episode_rewards = []
        episode_f1 = []
        episode_energy = []

        for w in range(n_windows):
            # recompute GT from current positions
            env.recompute_ground_truth()

            # build global state
            gs = self.build_global_state(obs)
            value = self.get_value(gs)

            # select actions for all agents
            actions, log_probs = self.select_actions(obs, gs)

            # run INDP window with these actions
            actions_per_slot = [actions] * cfg.N_slot
            window_result = protocol.run_window(nodes, cfg, rng, actions_per_slot)
            disc_adj = window_result["disc_adj"]
            env.set_discovered_topology(disc_adj)

            # compute F1 and energy
            gt = env.get_ground_truth_topology()
            f1_all, tp, fp, fn = protocol.compute_f1(gt, n)

            # counterfactual rewards
            active_ids = [nid for nid, st in protocol.states.items()
                          if st.tx_slots > 0]
            f1_without = self.compute_counterfactual_f1_batch(
                env, protocol, nodes, cfg, rng, active_ids, cfg.B_cf)

            energies = np.array([protocol.compute_energy(i, cfg) for i in range(n)],
                                dtype=np.float32)
            collisions = np.array(
                [protocol.states[i].collisions if i in protocol.states else 0
                 for i in range(n)], dtype=np.float32)

            rewards = self.compute_counterfactual_rewards(
                f1_all, f1_without, energies, collisions)

            done = (w == n_windows - 1)

            self.buffer.store(obs, gs, actions, log_probs, rewards,
                              done, np.full(n, value, dtype=np.float32))

            episode_rewards.append(float(rewards.mean()))
            episode_f1.append(f1_all)
            episode_energy.append(float(energies.mean()))

            # step env forward
            obs, _, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                obs, info = env.reset()

        # PPO update
        update_info = self.update()

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "mean_f1": float(np.mean(episode_f1)),
            "mean_energy": float(np.mean(episode_energy)),
            **update_info,
        }

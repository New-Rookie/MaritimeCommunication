"""
Experiment Block C — Learning-rate sweep for Improved IPPO.

Fix eta_N = 1.0, N_total = 120.
Train Improved IPPO at lr in {1e-4, 3e-4, 1e-3}.
Log per-episode mean reward.
Parallelized across (lr, seed) with ProcessPoolExecutor.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from P1.protocols.indp import INDPProtocol
from P1.algorithms.improved_ippo import ImprovedIPPO

LR_VALUES = [1e-4, 3e-4, 1e-3]
N_SEEDS = 1
N_EPISODES = 100
N_WINDOWS_PER_EP = 5


def _run_single_config_c(args):
    lr, seed, n_episodes, n_windows = args
    cfg = EnvConfig(N_total=50, eta_N=1.0, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="discovery", max_steps=n_windows * cfg.N_slot)
    rng = np.random.default_rng(seed)
    protocol = INDPProtocol(cfg)

    agent = ImprovedIPPO(
        n_agents=cfg.N_total, obs_dim=16, act_dim=2,
        lr=lr, cfg=cfg)

    records = []
    for ep in range(n_episodes):
        ep_info = agent.train_episode(env, protocol, n_windows=n_windows, rng=rng)
        records.append({
            "experiment": "C",
            "lr": lr,
            "seed": seed,
            "episode": ep,
            "mean_reward": ep_info["mean_reward"],
            "mean_f1": ep_info["mean_f1"],
            "mean_energy": ep_info["mean_energy"],
            "policy_loss": ep_info.get("policy_loss", 0),
            "value_loss": ep_info.get("value_loss", 0),
        })

    env.close()
    return records


def run_block_c(log_dir: str = "P1/logs", n_seeds: int = N_SEEDS,
                n_episodes: int = N_EPISODES,
                n_windows: int = N_WINDOWS_PER_EP,
                n_workers: int = None) -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    args_list = [(lr, seed, n_episodes, n_windows)
                 for lr in LR_VALUES
                 for seed in range(n_seeds)]

    all_records = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in tqdm(pool.map(_run_single_config_c, args_list),
                           total=len(args_list), desc="Block C (configs)",
                           unit="cfg", dynamic_ncols=True):
            all_records.extend(result)

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(log_dir, "block_c_raw.csv"), index=False)

    summary = df.groupby(["lr", "episode"])["mean_reward"].agg(
        ["mean", "std"]).reset_index()
    summary.to_csv(os.path.join(log_dir, "block_c_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_c()
    print(f"\nBlock C completed in {time.time() - t0:.1f}s")

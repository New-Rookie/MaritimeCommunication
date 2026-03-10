"""
Experiment Block A — GMAPPO learning-rate sweep  (parallelised).

Fix N_total = 120, eta_ch = 1.0.
Train GMAPPO at lr in {1e-4, 3e-4, 1e-3}.
Log per-episode mean reward and mean LA_pi.

Parallelisation granularity: one work-unit per (lr, seed) pair → 15 units.
Each worker independently creates EnvConfig / env / rng / estimator / agent.

Output: block_a_raw.csv, block_a_summary.csv
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv

from P2.link_quality.rf_estimator import LinkQualityEstimator
from P2.algorithms.gmappo import GMAPPO

LR_VALUES = [1e-4, 3e-4, 1e-3]
N_SEEDS = 5
N_EPISODES = 80
N_WINDOWS_PER_EP = 5


# ── top-level worker (picklable) ────────────────────────────────────────────

def _worker_block_a(
    args: Tuple[float, int, str | None, int, int],
) -> List[Dict[str, Any]]:
    """Train one (lr, seed) configuration and return per-episode records."""
    lr, seed, estimator_path, n_episodes, n_windows = args

    cfg = EnvConfig(N_total=30, eta_ch=1.0, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="link_selection",
                       max_steps=n_windows * 20 + 50)
    rng = np.random.default_rng(seed)

    estimator = LinkQualityEstimator()
    if estimator_path and os.path.exists(estimator_path):
        estimator.load(estimator_path)

    agent = GMAPPO(cfg.N_total, cfg, estimator, lr=lr)

    records: List[Dict[str, Any]] = []
    for ep in range(n_episodes):
        ep_info = agent.train_episode(env, n_windows=n_windows, rng=rng)
        records.append({
            "experiment": "A",
            "lr": lr,
            "seed": seed,
            "episode": ep,
            "mean_reward": ep_info["mean_reward"],
            "mean_LA": ep_info["mean_LA"],
            "policy_loss": ep_info.get("policy_loss", 0),
            "value_loss": ep_info.get("value_loss", 0),
        })

    env.close()
    return records


# ── public entry point ──────────────────────────────────────────────────────

def run_block_a(
    log_dir: str = "P2/logs",
    estimator: LinkQualityEstimator | None = None,
    estimator_path: str | None = None,
    n_seeds: int = N_SEEDS,
    n_episodes: int = N_EPISODES,
    n_windows: int = N_WINDOWS_PER_EP,
    n_workers: int | None = None,
) -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 32)

    # Resolve estimator path so every worker can load independently
    if estimator_path is None:
        default_path = os.path.join(log_dir, "rf_estimator.pkl")
        if estimator is not None and hasattr(estimator, "save"):
            estimator_path = os.path.join(log_dir, "_estimator_shared.pkl")
            estimator.save(estimator_path)
        elif os.path.exists(default_path):
            estimator_path = default_path

    work_units = [
        (lr, seed, estimator_path, n_episodes, n_windows)
        for lr in LR_VALUES
        for seed in range(n_seeds)
    ]

    records: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for batch in tqdm(
            pool.map(_worker_block_a, work_units),
            total=len(work_units),
            desc="Block A (configs)",
            unit="cfg",
            leave=True,
            dynamic_ncols=True,
        ):
            records.extend(batch)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(log_dir, "block_a_raw.csv"), index=False)

    summary = df.groupby(["lr", "episode"])["mean_reward"].agg(
        ["mean", "std"]).reset_index()
    summary.to_csv(os.path.join(log_dir, "block_a_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_a()
    print(f"\nBlock A completed in {time.time() - t0:.1f}s")

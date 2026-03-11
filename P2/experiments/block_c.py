"""
Experiment Block C — Algorithm comparison under channel-condition variation (parallelised).

Fix N_total = 20, sweep eta_ch in {0.75, 1.0, 1.25, 1.5}.
Compare GMAPPO / MAPPO / Greedy / ACO / GA.
Primary metric: mean LA_pi.

Parallelisation granularity: one work-unit per (eta_ch, seed, algorithm) → 200 units.
Each worker independently creates EnvConfig / env / rng / estimator / agent.

Output: block_c_raw.csv, block_c_summary.csv
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
from P2.algorithms.mappo import MAPPO
from P2.algorithms.greedy import GreedySelector
from P2.algorithms.aco import ACOSelector
from P2.algorithms.ga import GASelector

ETA_CH_VALUES = [0.75, 1.0, 1.25, 1.5]
N_SEEDS = 1
N_TRAIN_EPISODES = 30
N_EVAL_WINDOWS = 15
ALGO_NAMES = ["GMAPPO", "MAPPO", "Greedy", "ACO", "GA"]


# ── top-level helpers (picklable) ───────────────────────────────────────────

def _train_and_eval_rl(agent, env, n_train, n_eval, rng, n_windows_train=10):
    for _ in range(n_train):
        agent.train_episode(env, n_windows=n_windows_train, rng=rng)
    env.reset()
    las = []
    for _ in range(n_eval):
        result = agent.run_window(
            env,
            agent.path_mgr.select_source_buoys(
                env.nodes, agent.cfg.N_src, rng),
            n_steps=10, rng=rng,
        )
        las.append(result["mean_LA"])
    return float(np.mean(las))


def _worker_block_c(
    args: Tuple[float, int, str, str | None, int, int, str],
) -> List[Dict[str, Any]]:
    """Run one (eta_ch, seed, algorithm) configuration."""
    eta_ch, seed, algo_name, estimator_path, n_train, n_eval, n_windows_train, device = args

    cfg = EnvConfig(N_total=20, eta_ch=eta_ch, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="link_selection",
                       max_steps=n_eval * 20 + 100)
    rng = np.random.default_rng(seed)

    estimator = LinkQualityEstimator()
    if estimator_path and os.path.exists(estimator_path):
        estimator.load(estimator_path)

    n = 30

    if algo_name == "GMAPPO":
        agent = GMAPPO(n, cfg, estimator, lr=3e-4, device=device)
        mean_la = _train_and_eval_rl(agent, env, n_train, n_eval, rng, n_windows_train)
    elif algo_name == "MAPPO":
        agent = MAPPO(n, cfg, estimator, lr=3e-4, device=device)
        mean_la = _train_and_eval_rl(agent, env, n_train, n_eval, rng, n_windows_train)
    elif algo_name == "Greedy":
        agent = GreedySelector(n, cfg, estimator)
        result = agent.run_episode(env, n_eval, rng)
        mean_la = result["mean_LA"]
    elif algo_name == "ACO":
        agent = ACOSelector(n, cfg, estimator)
        result = agent.run_episode(env, n_eval, rng)
        mean_la = result["mean_LA"]
    else:  # GA
        agent = GASelector(n, cfg, estimator)
        result = agent.run_episode(env, n_eval, rng)
        mean_la = result["mean_LA"]

    env.close()

    return [{
        "experiment": "C",
        "eta_ch": eta_ch,
        "seed": seed,
        "algorithm": algo_name,
        "mean_LA": mean_la,
    }]


# ── public entry point ──────────────────────────────────────────────────────

def run_block_c(
    log_dir: str = "P2/logs",
    estimator: LinkQualityEstimator | None = None,
    estimator_path: str | None = None,
    n_seeds: int = N_SEEDS,
    n_train: int = N_TRAIN_EPISODES,
    n_eval: int = N_EVAL_WINDOWS,
    n_windows_train: int = 10,
    n_workers: int | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    if estimator_path is None:
        default_path = os.path.join(log_dir, "rf_estimator.pkl")
        if estimator is not None and hasattr(estimator, "save"):
            estimator_path = os.path.join(log_dir, "_estimator_shared.pkl")
            estimator.save(estimator_path)
        elif os.path.exists(default_path):
            estimator_path = default_path

    work_units = [
        (eta_ch, seed, algo_name, estimator_path, n_train, n_eval, n_windows_train, device)
        for eta_ch in ETA_CH_VALUES
        for seed in range(n_seeds)
        for algo_name in ALGO_NAMES
    ]

    records: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for batch in tqdm(
            pool.map(_worker_block_c, work_units),
            total=len(work_units),
            desc="Block C",
            unit="algo-run",
            leave=True,
            dynamic_ncols=True,
        ):
            records.extend(batch)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(log_dir, "block_c_raw.csv"), index=False)

    summary = df.groupby(["eta_ch", "algorithm"])["mean_LA"].agg(
        ["mean", "std"]).reset_index()
    summary.to_csv(os.path.join(log_dir, "block_c_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_c()
    print(summary.to_string(index=False))
    print(f"\nBlock C completed in {time.time() - t0:.1f}s")

"""
Experiment Block B — Algorithm comparison under node-count variation (parallelised).

Fix eta_ch = 1.0, sweep N_total in {40, 80, 120, 160, 200}.
Compare GMAPPO / MAPPO / Greedy / ACO / GA.
Primary metric: mean LA_pi.

Parallelisation granularity: one work-unit per (N_total, seed, algorithm) → 250 units.
Each worker independently creates EnvConfig / env / rng / estimator / agent.

Output: block_b_raw.csv, block_b_summary.csv
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

N_TOTAL_VALUES = [10, 15, 20, 25, 30]
N_SEEDS = 8
N_TRAIN_EPISODES = 30
N_EVAL_WINDOWS = 15
ALGO_NAMES = ["GMAPPO", "MAPPO", "Greedy", "ACO", "GA"]


# ── top-level helpers (picklable) ───────────────────────────────────────────

def _train_and_eval_rl(agent, env, n_train, n_eval, rng):
    for _ in range(n_train):
        agent.train_episode(env, n_windows=5, rng=rng)
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


def _worker_block_b(
    args: Tuple[int, int, str, str | None, int, int],
) -> List[Dict[str, Any]]:
    """Run one (N_total, seed, algorithm) configuration."""
    n_total, seed, algo_name, estimator_path, n_train, n_eval = args

    cfg = EnvConfig(N_total=n_total, eta_ch=1.0, print_diagnostics=False)
    env = MarineIoTEnv(cfg, mode="link_selection",
                       max_steps=n_eval * 20 + 100)
    rng = np.random.default_rng(seed)

    estimator = LinkQualityEstimator()
    if estimator_path and os.path.exists(estimator_path):
        estimator.load(estimator_path)

    n = n_total

    if algo_name == "GMAPPO":
        agent = GMAPPO(n, cfg, estimator, lr=3e-4)
        mean_la = _train_and_eval_rl(agent, env, n_train, n_eval, rng)
    elif algo_name == "MAPPO":
        agent = MAPPO(n, cfg, estimator, lr=3e-4)
        mean_la = _train_and_eval_rl(agent, env, n_train, n_eval, rng)
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
        "experiment": "B",
        "N_total": n_total,
        "seed": seed,
        "algorithm": algo_name,
        "mean_LA": mean_la,
    }]


# ── public entry point ──────────────────────────────────────────────────────

def run_block_b(
    log_dir: str = "P2/logs",
    estimator: LinkQualityEstimator | None = None,
    estimator_path: str | None = None,
    n_seeds: int = N_SEEDS,
    n_train: int = N_TRAIN_EPISODES,
    n_eval: int = N_EVAL_WINDOWS,
    n_workers: int | None = None,
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
        (n_total, seed, algo_name, estimator_path, n_train, n_eval)
        for n_total in N_TOTAL_VALUES
        for seed in range(n_seeds)
        for algo_name in ALGO_NAMES
    ]

    records: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for batch in tqdm(
            pool.map(_worker_block_b, work_units),
            total=len(work_units),
            desc="Block B",
            unit="algo-run",
            leave=True,
            dynamic_ncols=True,
        ):
            records.extend(batch)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(log_dir, "block_b_raw.csv"), index=False)

    summary = df.groupby(["N_total", "algorithm"])["mean_LA"].agg(
        ["mean", "std"]).reset_index()
    summary.to_csv(os.path.join(log_dir, "block_b_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_b()
    print(summary.to_string(index=False))
    print(f"\nBlock B completed in {time.time() - t0:.1f}s")

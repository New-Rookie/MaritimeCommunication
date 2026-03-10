"""
Experiment Block B — Average total delay under resource variation (parallelised).

Three separate sweeps: eta_B, eta_F, eta_S.
Only one scaling factor varies at a time; the other two remain at 1.0.
Compare: Improved MATD3, MATD3, Greedy, GA, ACO.

Block C (energy) is computed simultaneously — both metrics are logged
in the same run to avoid redundant computation.

Output: block_b_raw.csv, block_b_summary.csv
        block_c_raw.csv, block_c_summary.csv
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv

from P3.algorithms.improved_matd3 import ImprovedMATD3
from P3.algorithms.matd3 import MATD3
from P3.algorithms.greedy import GreedyAllocator
from P3.algorithms.aco import ACOAllocator
from P3.algorithms.ga import GAAllocator

ETA_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5]
ETA_TYPES = ["eta_B", "eta_F", "eta_S"]
N_SEEDS = 1
N_TRAIN = 70
N_EVAL = 20
ALGO_NAMES = ["Improved_MATD3", "MATD3", "Greedy", "ACO", "GA"]


def _make_cfg(eta_type: str, eta_val: float) -> EnvConfig:
    kw = {"N_total": 15, "print_diagnostics": False}
    if eta_type == "eta_B":
        kw["eta_B"] = eta_val
    elif eta_type == "eta_F":
        kw["eta_F"] = eta_val
    elif eta_type == "eta_S":
        kw["eta_S"] = eta_val
    return EnvConfig(**kw)


def _train_and_eval_rl(agent, env, n_train, n_eval, rng):
    for _ in range(n_train):
        agent.train_episode(env, n_windows=5, rng=rng)
    env.reset()
    T_vals, E_vals, G_vals = [], [], []
    for _ in range(n_eval):
        m = agent.eval_window(env, rng=rng)
        T_vals.append(m["mean_T_total"])
        E_vals.append(m["mean_E_total"])
        G_vals.append(m["mean_Gamma"])
    return float(np.mean(T_vals)), float(np.mean(E_vals)), float(np.mean(G_vals))


def _worker_block_bc(
    args: Tuple[str, float, int, str, int, int],
) -> List[Dict[str, Any]]:
    eta_type, eta_val, seed, algo_name, n_train, n_eval = args
    cfg = _make_cfg(eta_type, eta_val)
    env = MarineIoTEnv(cfg, mode="resource_mgmt",
                       max_steps=n_eval * 20 + 100)
    rng = np.random.default_rng(seed)
    n = min(cfg.N_src, cfg.node_counts["buoy"])

    if algo_name == "Improved_MATD3":
        agent = ImprovedMATD3(n, cfg, lr=3e-4)
        mean_T, mean_E, mean_G = _train_and_eval_rl(agent, env, n_train, n_eval, rng)
    elif algo_name == "MATD3":
        agent = MATD3(n, cfg, lr=3e-4)
        mean_T, mean_E, mean_G = _train_and_eval_rl(agent, env, n_train, n_eval, rng)
    elif algo_name == "Greedy":
        agent = GreedyAllocator(n, cfg)
        r = agent.run_episode(env, n_eval, rng)
        mean_T, mean_E, mean_G = r["mean_T_total"], r["mean_E_total"], r.get("mean_Gamma", 0.0)
    elif algo_name == "ACO":
        agent = ACOAllocator(n, cfg)
        r = agent.run_episode(env, n_eval, rng)
        mean_T, mean_E, mean_G = r["mean_T_total"], r["mean_E_total"], r.get("mean_Gamma", 0.0)
    else:  # GA
        agent = GAAllocator(n, cfg)
        r = agent.run_episode(env, n_eval, rng)
        mean_T, mean_E, mean_G = r["mean_T_total"], r["mean_E_total"], r.get("mean_Gamma", 0.0)

    env.close()
    return [{
        "experiment": "BC",
        "eta_type": eta_type,
        "eta_value": eta_val,
        "seed": seed,
        "algorithm": algo_name,
        "mean_T_total": mean_T,
        "mean_E_total": mean_E,
        "mean_Gamma": mean_G,
    }]


def run_block_bc(
    log_dir: str = "P3/logs",
    n_seeds: int = N_SEEDS,
    n_train: int = N_TRAIN,
    n_eval: int = N_EVAL,
    n_workers: int | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    work_units = [
        (eta_type, eta_val, seed, algo, n_train, n_eval)
        for eta_type in ETA_TYPES
        for eta_val in ETA_VALUES
        for seed in range(n_seeds)
        for algo in ALGO_NAMES
    ]

    records: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for batch in tqdm(
            pool.map(_worker_block_bc, work_units),
            total=len(work_units),
            desc="Block B+C (resource scaling)",
            unit="run",
            leave=True,
            dynamic_ncols=True,
        ):
            records.extend(batch)

    df = pd.DataFrame(records)

    # Block B: delay
    df_b = df[["eta_type", "eta_value", "seed", "algorithm", "mean_T_total"]].copy()
    df_b.insert(0, "experiment", "B")
    df_b.to_csv(os.path.join(log_dir, "block_b_raw.csv"), index=False)
    sum_b = df_b.groupby(["eta_type", "eta_value", "algorithm"])["mean_T_total"].agg(["mean", "std"]).reset_index()
    sum_b.to_csv(os.path.join(log_dir, "block_b_summary.csv"), index=False)

    # Block C: energy
    df_c = df[["eta_type", "eta_value", "seed", "algorithm", "mean_E_total"]].copy()
    df_c.insert(0, "experiment", "C")
    df_c.to_csv(os.path.join(log_dir, "block_c_raw.csv"), index=False)
    sum_c = df_c.groupby(["eta_type", "eta_value", "algorithm"])["mean_E_total"].agg(
        ["mean", "std"]).reset_index()
    sum_c.to_csv(os.path.join(log_dir, "block_c_summary.csv"), index=False)


    # Throughput under resource variation
    df_tp = df[["eta_type", "eta_value", "seed", "algorithm", "mean_Gamma"]].copy()
    df_tp.insert(0, "experiment", "B_throughput")
    df_tp.to_csv(os.path.join(log_dir, "block_b_throughput_raw.csv"), index=False)
    sum_tp = df_tp.groupby(["eta_type", "eta_value", "algorithm"])["mean_Gamma"].agg(["mean", "std"]).reset_index()
    sum_tp.to_csv(os.path.join(log_dir, "block_b_throughput_summary.csv"), index=False)

    return sum_b, sum_c


if __name__ == "__main__":
    t0 = time.time()
    sb, sc = run_block_bc()
    print(f"\nBlock B+C completed in {time.time() - t0:.1f}s")

"""
Experiment Block E — Algorithm comparison under node-count variation.

Fix eta_N = 1.0, sweep N_total over {20, 35, 50, 65, 80}.
Compare Improved IPPO / IPPO / Greedy / ACO / GA.
Primary metric: mean E_ND.
Parallelized across (N_total, seed, algorithm) with ProcessPoolExecutor.
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
from P1.algorithms.ippo import IPPO
from P1.algorithms.greedy import GreedyOptimizer
from P1.algorithms.aco import ACOOptimizer
from P1.algorithms.ga import GAOptimizer

N_TOTAL_VALUES = [20, 35, 50, 65, 80]
N_SEEDS = 1
N_TRAIN_EPISODES = 40
N_EVAL_WINDOWS = 10

ALGO_NAMES = ["Improved_IPPO", "IPPO", "Greedy", "ACO", "GA"]


def _train_and_eval_rl(agent, env, protocol, cfg, rng, n_train, n_eval, n_windows_train=10):
    for ep in range(n_train):
        agent.train_episode(env, protocol, n_windows=n_windows_train, rng=rng)

    obs, _ = env.reset()
    nodes = env.nodes
    n = len(nodes)
    energies = []
    for w in range(n_eval):
        if hasattr(agent, "build_global_state"):
            gs = agent.build_global_state(obs)
            actions, _ = agent.select_actions(obs, gs)
        else:
            actions, _ = agent.select_actions(obs)
        result = protocol.run_window(nodes, cfg, rng, [actions] * cfg.N_slot)
        energies.append(result["mean_energy"])
        obs, _, term, trunc, _ = env.step(actions)
        if term or trunc:
            obs, _ = env.reset()
    return float(np.mean(energies))


def _run_single_config_e(args):
    n_total, seed, algo_name, n_train, n_eval, n_windows_train, device = args
    cfg = EnvConfig(N_total=n_total, eta_N=1.0, print_diagnostics=False)
    rng = np.random.default_rng(seed)
    n = n_total

    algo_factories = {
        "Improved_IPPO": lambda: ImprovedIPPO(n, cfg=cfg, lr=3e-4, device=device),
        "IPPO": lambda: IPPO(n, cfg=cfg, lr=3e-4, device=device),
        "Greedy": lambda: GreedyOptimizer(n, cfg=cfg),
        "ACO": lambda: ACOOptimizer(n, cfg=cfg),
        "GA": lambda: GAOptimizer(n, cfg=cfg),
    }

    env = MarineIoTEnv(cfg, mode="discovery", max_steps=n_eval * cfg.N_slot)
    protocol = INDPProtocol(cfg)
    agent = algo_factories[algo_name]()

    if algo_name in ("Improved_IPPO", "IPPO"):
        mean_e = _train_and_eval_rl(agent, env, protocol, cfg, rng, n_train, n_eval, n_windows_train)
    else:
        result = agent.run_episode(env, protocol, n_eval, rng)
        mean_e = result["mean_energy"]

    env.close()
    return [{
        "experiment": "E",
        "N_total": n_total,
        "seed": seed,
        "algorithm": algo_name,
        "mean_E_ND": mean_e,
    }]


def run_block_e(log_dir: str = "P1/logs", n_seeds: int = N_SEEDS,
                n_train: int = N_TRAIN_EPISODES,
                n_eval: int = N_EVAL_WINDOWS,
                n_windows_train: int = 10,
                n_workers: int = None,
                device: str = "cpu") -> pd.DataFrame:
    os.makedirs(log_dir, exist_ok=True)
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 48)

    args_list = [(n_total, seed, algo_name, n_train, n_eval, n_windows_train, device)
                 for n_total in N_TOTAL_VALUES
                 for seed in range(n_seeds)
                 for algo_name in ALGO_NAMES]

    all_records = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for result in tqdm(pool.map(_run_single_config_e, args_list),
                           total=len(args_list), desc="Block E",
                           unit="algo-run", dynamic_ncols=True):
            all_records.extend(result)

    df = pd.DataFrame(all_records)
    df.to_csv(os.path.join(log_dir, "block_e_raw.csv"), index=False)

    summary = df.groupby(["N_total", "algorithm"])["mean_E_ND"].agg(
        ["mean", "std"]).reset_index()
    summary.to_csv(os.path.join(log_dir, "block_e_summary.csv"), index=False)
    return summary


if __name__ == "__main__":
    t0 = time.time()
    summary = run_block_e()
    print(summary.to_string(index=False))
    print(f"\nBlock E completed in {time.time() - t0:.1f}s")

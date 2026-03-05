"""
P2 – Experiment 2: Different channel conditions (fixed topology).

Compare: MADDPG, MAPPO, Greedy, ACO, MAPPO+GCN
Metrics: (1) Link switching stability  (2) Communication delay  (3) Energy

All algorithms train in parallel via multiprocessing.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from P2.experiments.run_exp1 import (
    ALGO_NAMES, STYLES, RESULTS_DIR,
    _pool_worker_exp1, _init_pool_worker,
)

TRAIN_EPISODES = 800
EVAL_EPISODES = 20
EP_LEN = 200
FIXED_COUNTS = {"satellite": 3, "uav": 6, "ship": 10,
                "buoy": 20, "base_station": 3}
NOISE_LEVELS = [0.5, 1.0, 2.0, 4.0, 7.0, 10.0]


def _save_exp2_xlsx(all_results, algo_names, noise_levels):
    try:
        from openpyxl import Workbook
    except ImportError:
        print("  [WARN] openpyxl not installed, skipping xlsx save")
        return

    for metric in ["stability", "delay", "energy"]:
        wb = Workbook()
        ws = wb.active
        ws.title = metric
        ws.append(["Noise_Factor"] + algo_names)
        for i, nf in enumerate(noise_levels):
            row = [nf]
            for algo in algo_names:
                row.append(all_results[algo][metric][i])
            ws.append(row)
        fname = f"exp2_{metric}_data.xlsx"
        wb.save(os.path.join(RESULTS_DIR, fname))
        print(f"  Saved {fname}")


def run_exp2(train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES,
             ep_len=EP_LEN, node_counts=None, noise_levels=None,
             max_workers=5, gpu_frac=0.15,
             save_xlsx=True, save_plots=True):
    if node_counts is None:
        node_counts = FIXED_COUNTS
    if noise_levels is None:
        noise_levels = NOISE_LEVELS
    algo_names = ALGO_NAMES

    all_results = {a: {"delay": [], "energy": [], "stability": []}
                   for a in algo_names}

    tasks = []
    for nf in noise_levels:
        for algo in algo_names:
            tasks.append((algo, node_counts, nf,
                          train_episodes, eval_episodes, ep_len))

    print(f"\n=== P2 Exp-2: Varying channel ({len(tasks)} tasks, "
          f"{max_workers} workers) ===")

    if max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx,
                                 initializer=_init_pool_worker,
                                 initargs=(gpu_frac,)) as pool:
            future_map = {}
            for i, t in enumerate(tasks):
                future_map[pool.submit(_pool_worker_exp1, t)] = i
            results = [None] * len(tasks)
            for f in tqdm(as_completed(future_map), total=len(tasks),
                          desc="P2-Exp2"):
                results[future_map[f]] = f.result()
    else:
        results = []
        for t in tqdm(tasks, desc="P2-Exp2"):
            results.append(_pool_worker_exp1(t))

    idx = 0
    for _ in noise_levels:
        for algo in algo_names:
            m = results[idx]
            for k in ["delay", "energy", "stability"]:
                all_results[algo][k].append(m[k])
            idx += 1

    if save_xlsx:
        _save_exp2_xlsx(all_results, algo_names, noise_levels)

    if save_plots:
        metric_labels = {
            "stability": ("Link Switching Stability", "exp2_1_stability.png"),
            "delay": ("Communication Delay (s)", "exp2_2_delay.png"),
            "energy": ("Communication Energy (J)", "exp2_3_energy.png"),
        }
        for metric, (ylabel, fname) in metric_labels.items():
            plt.figure(figsize=(8, 5))
            for algo in algo_names:
                plt.plot(noise_levels, all_results[algo][metric],
                         STYLES[algo], label=algo, linewidth=2, markersize=7)
            plt.xlabel("Noise Factor", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.title(f"P2 Exp-2: {ylabel} vs. Channel Condition",
                      fontsize=13)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=200)
            plt.close()
            print(f"  Saved {fname}")

    return all_results


if __name__ == "__main__":
    run_exp2()

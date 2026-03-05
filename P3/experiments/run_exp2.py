"""
P3 – Experiment 2: Varying environmental noise (fixed resources).

3 sub-experiments: delay / energy / throughput vs. noise factor

Compare: MADDPG, MATD3, Greedy, GA, IMATD3

All algorithms train in parallel via multiprocessing.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from P3.experiments.run_exp1 import (
    ALGO_NAMES, STYLES, RESULTS_DIR,
    _pool_worker_exp1, _init_pool_worker,
    TRAIN_EPISODES, EVAL_EPISODES, EP_LEN,
)

NOISE_LEVELS = [0.5, 1.0, 2.0, 4.0, 7.0, 10.0]


def _save_exp2_xlsx(all_results, algo_names, noise_levels):
    try:
        from openpyxl import Workbook
    except ImportError:
        print("  [WARN] openpyxl not installed, skipping xlsx save")
        return

    for metric in ["delay", "energy", "throughput"]:
        wb = Workbook()
        ws = wb.active
        ws.title = metric
        ws.append(["Noise_Factor"] + algo_names)
        for i, nf in enumerate(noise_levels):
            row = [nf]
            for algo in algo_names:
                row.append(all_results[algo][metric][i])
            ws.append(row)
        fname = f"exp2_noise_{metric}_data.xlsx"
        wb.save(os.path.join(RESULTS_DIR, fname))
        print(f"  Saved {fname}")


def run_exp2(train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES,
             ep_len=EP_LEN, noise_levels=None,
             max_workers=5, gpu_frac=0.15,
             save_xlsx=True, save_plots=True):
    if noise_levels is None:
        noise_levels = NOISE_LEVELS

    metrics_list = ["delay", "energy", "throughput"]
    metric_labels = {
        "delay": "Total Delay (s)",
        "energy": "Total Energy (J)",
        "throughput": "Avg Throughput (bytes/s)",
    }
    all_results = {a: {m: [] for m in metrics_list} for a in ALGO_NAMES}

    tasks = []
    for nf in noise_levels:
        env_kwargs = dict(noise_factor=nf, episode_length=ep_len)
        for algo in ALGO_NAMES:
            tasks.append((algo, env_kwargs, train_episodes,
                          eval_episodes, ep_len))

    print(f"\n=== P3 Exp-2: Varying noise ({len(tasks)} tasks, "
          f"{max_workers} workers) ===")

    if max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=ctx,
            initializer=_init_pool_worker, initargs=(gpu_frac,)
        ) as pool:
            fmap = {}
            for i, t in enumerate(tasks):
                fmap[pool.submit(_pool_worker_exp1, t)] = i
            results = [None] * len(tasks)
            for f in tqdm(as_completed(fmap), total=len(tasks),
                          desc="P3-Exp2"):
                results[fmap[f]] = f.result()
    else:
        results = []
        for t in tqdm(tasks, desc="P3-Exp2"):
            results.append(_pool_worker_exp1(t))

    idx = 0
    for _ in noise_levels:
        for algo in ALGO_NAMES:
            m = results[idx]
            for k in metrics_list:
                all_results[algo][k].append(m[k])
            idx += 1

    if save_xlsx:
        _save_exp2_xlsx(all_results, ALGO_NAMES, noise_levels)

    if save_plots:
        for i, metric in enumerate(metrics_list, 1):
            plt.figure(figsize=(8, 5))
            for algo in ALGO_NAMES:
                plt.plot(noise_levels, all_results[algo][metric],
                         STYLES[algo], label=algo, linewidth=2, markersize=7)
            plt.xlabel("Environmental Noise Factor", fontsize=12)
            plt.ylabel(metric_labels[metric], fontsize=12)
            plt.title(f"P3 Exp-2-{i}: {metric_labels[metric]} vs. Noise",
                      fontsize=13)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = f"exp2_{i}_noise_{metric}.png"
            plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=200)
            plt.close()
            print(f"  Saved {fname}")

    return all_results


if __name__ == "__main__":
    run_exp2()

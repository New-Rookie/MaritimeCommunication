"""
Master experiment runner — orchestrates all experiment blocks
for Research Content 2 (MEC-aware link selection).

Usage:
    python -m P2.experiments.runner [--blocks P A B C] [--quick] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from P2.experiments.block_p import run_block_p
from P2.experiments.block_a import run_block_a
from P2.experiments.block_b import run_block_b
from P2.experiments.block_c import run_block_c
from P2.experiments.block_d import run_block_d
from P2.link_quality.rf_estimator import LinkQualityEstimator


BLOCK_MAP = {
    "P": "run_block_p",
    "A": "run_block_a",
    "B": "run_block_b",
    "C": "run_block_c",
    "D": "run_block_d",
}

BLOCK_DESC = {
    "P": "Offline RF estimator (effectiveness chart)",
    "A": "GMAPPO learning-rate sweep (reward curve)",
    "B": "Algorithm comparison vs N_total (LA chart)",
    "C": "Algorithm comparison vs eta_ch (LA chart)",
    "D": "Convergence comparison (GMAPPO vs MAPPO)",
}


def _resolve_execution_config(device_arg: str, cpu_cores: int | None, cpu_utilization: float) -> tuple[str, int, int]:
    cpu_total = os.cpu_count() or 1
    cpu_budget = max(1, min(cpu_total, cpu_cores if cpu_cores is not None else int(round(cpu_total * cpu_utilization))))

    device = "cpu"
    if device_arg in ("cpu", "cuda"):
        device = device_arg
    else:
        import importlib.util
        import importlib
        has_torch = importlib.util.find_spec("torch") is not None
        if has_torch:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                device = "cuda"

    # concurrent P1/P2/P3 default split unless explicit cpu_cores given
    if cpu_cores is None:
        worker_budget = max(1, cpu_budget // 3)
    else:
        worker_budget = cpu_budget

    # GPU mode: avoid oversubscribing a single accelerator by default
    if device == "cuda":
        worker_budget = max(1, min(worker_budget, 4))

    return device, cpu_budget, worker_budget


def main():
    parser = argparse.ArgumentParser(
        description="RC2 Link Selection Experiment Runner")
    parser.add_argument("--blocks", nargs="*",
                        default=list(BLOCK_MAP.keys()),
                        help="Blocks to run (default: all)")
    parser.add_argument("--log-dir", default="P2/logs",
                        help="Output directory for CSV logs")
    parser.add_argument("--quick", action="store_true",
                        help="Reduce seeds/episodes for smoke test")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (overrides auto budget)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Execution device preference for RL blocks")
    parser.add_argument("--cpu-cores", type=int, default=None,
                        help="CPU cores budget for this runner")
    parser.add_argument("--cpu-utilization", type=float, default=1.0,
                        help="Fraction of local CPU to budget when --cpu-cores is not set")
    parser.add_argument("--rl-episodes", type=int, default=None,
                        help="Override RL training episodes for episodic RL blocks")
    parser.add_argument("--rl-windows", type=int, default=10,
                        help="Training timesteps/windows per RL episode")
    args = parser.parse_args()

    device, cpu_budget, auto_workers = _resolve_execution_config(
        args.device, args.cpu_cores, args.cpu_utilization)
    resolved_workers = args.workers if args.workers is not None else auto_workers

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("P2/figures", exist_ok=True)

    blocks = [b.upper() for b in args.blocks if b.upper() in BLOCK_MAP]
    if not blocks:
        print("No valid blocks specified.")
        return

    worker_kw = {"n_workers": resolved_workers}

    print()
    print("=" * 64)
    print("  Research Content 2 — Link Selection Experiment Campaign")
    print("=" * 64)
    if args.quick:
        print("  MODE: --quick (reduced seeds/episodes for smoke test)")
    print(f"  Blocks: {', '.join(blocks)}")
    print(f"  Log dir: {args.log_dir}/")
    print(f"  Device: {device}")
    print(f"  CPU budget: {cpu_budget} cores")
    print(f"  Workers: {resolved_workers} ({'manual' if args.workers is not None else 'auto'})")
    print("=" * 64)
    print()

    total_t0 = time.time()
    estimator = None
    estimator_path = os.path.join(args.log_dir, "rf_estimator.pkl")

    pbar_blocks = tqdm(blocks, desc="Overall", unit="block",
                       leave=True, dynamic_ncols=True)

    for block_id in pbar_blocks:
        desc = BLOCK_DESC.get(block_id, "")
        pbar_blocks.set_postfix_str(f"Block {block_id}: {desc}")

        print(f"\n{'─' * 64}")
        print(f"  Block {block_id} — {desc}")
        print(f"{'─' * 64}")
        t0 = time.time()

        if block_id == "P":
            kw = {}
            if args.quick:
                kw["n_probe"] = 2000
                kw["n_steps"] = 100
            result = run_block_p(log_dir=args.log_dir, **kw)
            estimator = result["estimator"]

        elif block_id == "A":
            if estimator is None:
                estimator = _load_estimator(args.log_dir)
            kw = {"estimator": estimator,
                  "estimator_path": estimator_path,
                  "device": device,
                  **worker_kw}
            if args.quick:
                kw["n_seeds"] = 2
                kw["n_episodes"] = 10
                kw["n_windows"] = 3
            if args.rl_episodes is not None:
                kw["n_episodes"] = args.rl_episodes
            kw["n_windows"] = args.rl_windows
            run_block_a(log_dir=args.log_dir, **kw)

        elif block_id == "B":
            if estimator is None:
                estimator = _load_estimator(args.log_dir)
            kw = {"estimator": estimator,
                  "estimator_path": estimator_path,
                  "device": device,
                  **worker_kw}
            if args.quick:
                kw["n_seeds"] = 2
                kw["n_train"] = 10
                kw["n_windows_train"] = 3
            if args.rl_episodes is not None:
                kw["n_train"] = args.rl_episodes
            kw["n_windows_train"] = args.rl_windows
            run_block_b(log_dir=args.log_dir, **kw)

        elif block_id == "C":
            if estimator is None:
                estimator = _load_estimator(args.log_dir)
            kw = {"estimator": estimator,
                  "estimator_path": estimator_path,
                  "device": device,
                  **worker_kw}
            if args.quick:
                kw["n_seeds"] = 2
                kw["n_train"] = 10
                kw["n_windows_train"] = 3
            if args.rl_episodes is not None:
                kw["n_train"] = args.rl_episodes
            kw["n_windows_train"] = args.rl_windows
            run_block_c(log_dir=args.log_dir, **kw)


        elif block_id == "D":
            if estimator is None:
                estimator = _load_estimator(args.log_dir)
            kw = {"estimator_path": estimator_path, "device": device, **worker_kw}
            if args.quick:
                kw["n_seeds"] = 2
                kw["n_episodes"] = 10
                kw["n_windows"] = 3
            if args.rl_episodes is not None:
                kw["n_episodes"] = args.rl_episodes
            kw["n_windows"] = args.rl_windows
            run_block_d(log_dir=args.log_dir, **kw)

        elapsed = time.time() - t0
        print(f"  Block {block_id} completed in {elapsed:.1f}s")

    pbar_blocks.close()

    total = time.time() - total_t0
    print(f"\n{'=' * 64}")
    print(f"  All blocks completed in {total:.1f}s")
    print(f"  Logs saved to: {args.log_dir}/")
    print(f"{'=' * 64}")

    try:
        from P2.plotting.plot_results import generate_all_figures
        generate_all_figures(log_dir=args.log_dir, fig_dir="P2/figures")
        print("  Figures saved to: P2/figures/")
    except Exception as e:
        print(f"  Plot generation failed: {e}")


def _load_estimator(log_dir: str) -> LinkQualityEstimator:
    est = LinkQualityEstimator()
    mp = os.path.join(log_dir, "rf_estimator.pkl")
    if os.path.exists(mp):
        est.load(mp)
        print(f"  Loaded RF estimator from {mp}")
    else:
        print("  WARNING: No pre-trained RF estimator found. "
              "Run Block P first.")
    return est


if __name__ == "__main__":
    main()

"""
Master experiment runner — orchestrates all experiment blocks
for Research Content 3 (MEC Resource Management with Improved MATD3).

Usage:
    python -m P3.experiments.runner [--blocks A B C D] [--quick] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from P3.experiments.block_a import run_block_a
from P3.experiments.block_b import run_block_bc
from P3.experiments.block_d import run_block_d
from P3.experiments.block_e import run_block_e


BLOCK_MAP = {
    "A": "run_block_a",
    "B": "run_block_bc",
    "C": "run_block_bc",
    "D": "run_block_d",
    "E": "run_block_e",
}

BLOCK_DESC = {
    "A": "Improved MATD3 learning-rate sweep (reward curve)",
    "B": "Average delay under resource scaling (eta_B/eta_F/eta_S)",
    "C": "Average energy under resource scaling (eta_B/eta_F/eta_S)",
    "D": "Delay + Energy + Throughput under data-volume variation (M_tot sweep)",
    "E": "Convergence comparison (Improved MATD3 vs MATD3)",
}


def main():
    parser = argparse.ArgumentParser(
        description="RC3 Resource Management Experiment Runner")
    parser.add_argument("--blocks", nargs="*",
                        default=["A", "B", "C", "D", "E"],
                        help="Blocks to run (default: all)")
    parser.add_argument("--log-dir", default="P3/logs",
                        help="Output directory for CSV logs")
    parser.add_argument("--quick", action="store_true",
                        help="Reduce seeds/episodes for smoke test")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: cpu_count)")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("P3/figures", exist_ok=True)

    requested = [b.upper() for b in args.blocks]
    # B and C share a single run; deduplicate
    blocks: list[str] = []
    bc_done = False
    for b in requested:
        if b in ("B", "C"):
            if not bc_done:
                blocks.append("BC")
                bc_done = True
        elif b in BLOCK_MAP:
            blocks.append(b)

    if not blocks:
        print("No valid blocks specified.")
        return

    worker_kw = {"n_workers": args.workers}

    print()
    print("=" * 64)
    print("  Research Content 3 — Resource Management Experiment Campaign")
    print("=" * 64)
    if args.quick:
        print("  MODE: --quick (reduced seeds/episodes for smoke test)")
    print(f"  Blocks: {', '.join(requested)}")
    print(f"  Log dir: {args.log_dir}/")
    print(f"  Workers: {args.workers or 'auto (cpu_count)'}")
    print("=" * 64)
    print()

    total_t0 = time.time()

    pbar = tqdm(blocks, desc="Overall", unit="block",
                leave=True, dynamic_ncols=True)

    for block_id in pbar:
        if block_id == "BC":
            desc = "Blocks B+C — delay + energy under resource scaling"
        else:
            desc = BLOCK_DESC.get(block_id, "")
        pbar.set_postfix_str(desc)

        print(f"\n{'─' * 64}")
        print(f"  {desc}")
        print(f"{'─' * 64}")
        t0 = time.time()

        if block_id == "A":
            kw = {**worker_kw}
            if args.quick:
                kw["n_seeds"] = 2
                kw["n_episodes"] = 10
            run_block_a(log_dir=args.log_dir, **kw)

        elif block_id == "BC":
            kw = {**worker_kw}
            if args.quick:
                kw["n_seeds"] = 2
                kw["n_train"] = 10
                kw["n_eval"] = 3
            run_block_bc(log_dir=args.log_dir, **kw)

        elif block_id == "D":
            kw = {**worker_kw}
            if args.quick:
                kw["n_seeds"] = 2
                kw["n_train"] = 10
                kw["n_eval"] = 3
            run_block_d(log_dir=args.log_dir, **kw)

        elif block_id == "E":
            kw = {**worker_kw}
            if args.quick:
                kw["n_seeds"] = 2
                kw["n_episodes"] = 10
            run_block_e(log_dir=args.log_dir, **kw)

        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

    pbar.close()

    total = time.time() - total_t0
    print(f"\n{'=' * 64}")
    print(f"  All blocks completed in {total:.1f}s")
    print(f"  Logs saved to: {args.log_dir}/")
    print(f"{'=' * 64}")

    try:
        from P3.plotting.plot_results import generate_all_figures
        generate_all_figures(log_dir=args.log_dir, fig_dir="P3/figures")
        print("  Figures saved to: P3/figures/")
    except Exception as e:
        print(f"  Plot generation failed: {e}")


if __name__ == "__main__":
    main()

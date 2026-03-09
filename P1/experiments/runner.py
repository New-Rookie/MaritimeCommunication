"""
Master experiment runner — orchestrates all five experiment blocks
for Research Content 1 (INDP neighbour discovery).

Usage:
    python -m P1.experiments.runner [--blocks A B C D E] [--quick] [--workers N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from P1.experiments.block_a import run_block_a
from P1.experiments.block_b import run_block_b
from P1.experiments.block_c import run_block_c
from P1.experiments.block_d import run_block_d
from P1.experiments.block_e import run_block_e


BLOCK_MAP = {
    "A": run_block_a,
    "B": run_block_b,
    "C": run_block_c,
    "D": run_block_d,
    "E": run_block_e,
}

BLOCK_DESC = {
    "A": "Mechanism comparison vs noise (F1_topo)",
    "B": "Mechanism comparison vs node count (F1_topo)",
    "C": "Learning-rate sweep (Improved IPPO reward curve)",
    "D": "Algorithm comparison vs noise (E_ND)",
    "E": "Algorithm comparison vs node count (E_ND)",
}


def main():
    parser = argparse.ArgumentParser(description="RC1 INDP Experiment Runner")
    parser.add_argument("--blocks", nargs="*", default=list(BLOCK_MAP.keys()),
                        help="Blocks to run (default: all)")
    parser.add_argument("--log-dir", default="P1/logs",
                        help="Output directory for CSV logs")
    parser.add_argument("--quick", action="store_true",
                        help="Reduce seeds/episodes for a fast smoke test")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: cpu_count)")
    args = parser.parse_args()

    quick_kw = {}
    if args.quick:
        quick_kw = {"n_seeds": 2}

    worker_kw = {"n_workers": args.workers}

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("P1/figures", exist_ok=True)

    blocks_to_run = [b.upper() for b in args.blocks if b.upper() in BLOCK_MAP]
    if not blocks_to_run:
        print("No valid blocks specified.")
        return

    print()
    print("=" * 64)
    print("  Research Content 1 — INDP Experiment Campaign")
    print("=" * 64)
    if args.quick:
        print("  MODE: --quick (reduced seeds/episodes for smoke test)")
    print(f"  Blocks: {', '.join(blocks_to_run)}")
    print(f"  Log dir: {args.log_dir}/")
    print(f"  Workers: {args.workers or 'auto (cpu_count)'}")
    print("=" * 64)
    print()

    total_t0 = time.time()

    pbar_blocks = tqdm(blocks_to_run, desc="Overall", unit="block",
                       leave=True, dynamic_ncols=True)

    for block_id in pbar_blocks:
        desc = BLOCK_DESC.get(block_id, "")
        pbar_blocks.set_postfix_str(f"Block {block_id}: {desc}")

        print(f"\n{'─' * 64}")
        print(f"  Block {block_id} — {desc}")
        print(f"{'─' * 64}")
        t0 = time.time()

        if block_id == "C":
            kw = {**quick_kw, **worker_kw}
            if args.quick:
                kw["n_episodes"] = 10
                kw["n_seeds"] = 2
            BLOCK_MAP[block_id](log_dir=args.log_dir, **kw)
        elif block_id in ("D", "E"):
            kw = {**quick_kw, **worker_kw}
            if args.quick:
                kw["n_train"] = 10
            BLOCK_MAP[block_id](log_dir=args.log_dir, **kw)
        else:
            BLOCK_MAP[block_id](log_dir=args.log_dir, **quick_kw, **worker_kw)

        elapsed = time.time() - t0
        print(f"  Block {block_id} completed in {elapsed:.1f}s")

    pbar_blocks.close()

    total = time.time() - total_t0
    print(f"\n{'=' * 64}")
    print(f"  All blocks completed in {total:.1f}s")
    print(f"  Logs saved to: {args.log_dir}/")
    print(f"{'=' * 64}")

    try:
        from P1.plotting.plot_results import generate_all_figures
        generate_all_figures(log_dir=args.log_dir, fig_dir="P1/figures")
        print("  Figures saved to: P1/figures/")
    except Exception as e:
        print(f"  Plot generation failed: {e}")


if __name__ == "__main__":
    main()

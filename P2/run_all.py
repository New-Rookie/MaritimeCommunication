"""Run all P2 (Link Selection) experiments."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from P2.experiments.run_exp1 import run_exp1
from P2.experiments.run_exp2 import run_exp2

if __name__ == "__main__":
    print("=" * 60)
    print("  P2: Link Selection Experiments")
    print("=" * 60)

    run_exp1()
    run_exp2()

    print("\n" + "=" * 60)
    print("  All P2 experiments complete.  Results in P2/results/")
    print("=" * 60)

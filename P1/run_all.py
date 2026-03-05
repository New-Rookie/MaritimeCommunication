"""Run all P1 (Neighbour Discovery) experiments."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from P1.experiments.run_exp1 import run_exp1
from P1.experiments.run_exp2 import run_exp2_1, run_exp2_2_to_5

if __name__ == "__main__":
    print("=" * 60)
    print("  P1: Neighbour Discovery Experiments")
    print("=" * 60)

    print("\n--- Experiment 1: Mechanism comparison ---")
    run_exp1()

    print("\n--- Experiment 2: Algorithm comparison (INDP) ---")
    run_exp2_1()
    run_exp2_2_to_5()

    print("\n" + "=" * 60)
    print("  All P1 experiments complete.  Results in P1/results/")
    print("=" * 60)

"""Run all P3 (Resource Management) experiments."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from P3.experiments.run_exp1 import run_exp1
from P3.experiments.run_exp2 import run_exp2
from P3.experiments.run_exp3 import run_exp3

if __name__ == "__main__":
    print("=" * 60)
    print("  P3: Resource Management Experiments")
    print("=" * 60)

    run_exp1()
    run_exp2()
    run_exp3()

    print("\n" + "=" * 60)
    print("  All P3 experiments complete.  Results in P3/results/")
    print("=" * 60)

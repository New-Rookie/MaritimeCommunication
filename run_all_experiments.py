"""
Master script: run ALL experiments (P1 + P2 + P3) **in parallel**.

Each phase runs as a separate subprocess.
Total wall-clock time ~ max(T_P1, T_P2, T_P3) instead of their sum.

Usage:
    python run_all_experiments.py

For fine-grained parameter control, run the individual scripts:
    python run_p1.py
    python run_p2.py
    python run_p3.py
"""

import sys
import os
import time
import subprocess

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
os.environ["SDL_VIDEODRIVER"] = "dummy"


def main():
    print("=" * 70)
    print("  SAGIN Ocean IoT MEC – Full Experiment Suite")
    print("  Mode: 3 parallel subprocesses (P1 | P2 | P3)")
    print("=" * 70)

    t0 = time.time()
    python = sys.executable

    procs = {}
    for name, script in [("P1", "run_p1.py"), ("P2", "run_p2.py"),
                          ("P3", "run_p3.py")]:
        script_path = os.path.join(_ROOT, script)
        p = subprocess.Popen(
            [python, script_path],
            cwd=_ROOT,
            env={**os.environ, "PYTHONPATH": _ROOT},
        )
        procs[name] = p
        print(f"  [{name}] Started (PID {p.pid})")

    for name, p in procs.items():
        p.wait()
        if p.returncode != 0:
            print(f"  WARNING: {name} exited with code {p.returncode}")
        else:
            print(f"  [{name}] COMPLETE")

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print(f"  Total wall-clock time: {elapsed/60:.1f} min")
    print("  Results:")
    print("    P1/results/  – Neighbour discovery plots + xlsx")
    print("    P2/results/  – Link selection plots + xlsx")
    print("    P3/results/  – Resource management plots + xlsx")
    print("=" * 70)


if __name__ == "__main__":
    main()

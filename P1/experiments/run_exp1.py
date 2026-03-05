"""
P1 – Experiment 1: Discovery accuracy comparison across mechanisms.

(1) Fixed node count, varying environmental noise -> accuracy
(2) Fixed noise, varying node count -> accuracy

Mechanisms compared: INDP, Disco, ALOHA
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from env.ocean_env import OceanEnv
from env.config import SINR_THRESHOLD_DB
from P1.mechanisms.indp import INDPMechanism
from P1.mechanisms.disco import DiscoMechanism
from P1.mechanisms.aloha import ALOHAMechanism

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_accuracy(discovered, ground_truth_adj):
    """F1-based accuracy: harmonic mean of precision and recall."""
    total_tp = 0
    total_disc = 0
    total_true = 0
    for nid, true_nb in ground_truth_adj.items():
        true_set = set(true_nb.keys())
        disc_set = discovered.get(nid, set())
        total_tp += len(disc_set & true_set)
        total_disc += len(disc_set)
        total_true += len(true_set)
    precision = total_tp / max(total_disc, 1)
    recall = total_tp / max(total_true, 1)
    if precision + recall < 1e-10:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _save_exp1_xlsx(noise_factors, total_nodes_list, results_noise,
                    results_nodes, mech_names):
    try:
        from openpyxl import Workbook
    except ImportError:
        print("  [WARN] openpyxl not installed, skipping xlsx save")
        return

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Exp1_1_Noise"
    ws1.append(["Noise_Factor"] + mech_names)
    for i, nf in enumerate(noise_factors):
        row = [nf]
        for name in mech_names:
            row.append(results_noise[name][i])
        ws1.append(row)

    ws2 = wb.create_sheet("Exp1_2_Nodes")
    ws2.append(["Total_Nodes"] + mech_names)
    for i, tn in enumerate(total_nodes_list):
        row = [tn]
        for name in mech_names:
            row.append(results_nodes[name][i])
        ws2.append(row)

    wb.save(os.path.join(RESULTS_DIR, "exp1_mechanism_data.xlsx"))
    print("  Saved exp1_mechanism_data.xlsx")


def run_exp1_1(noise_factors=None, n_trials=5, save_plots=True):
    """Fixed node count, varying noise -> accuracy."""
    if noise_factors is None:
        noise_factors = [0.0, 1.0, 2.0, 4.0, 7.0, 10.0]

    print("\n=== Exp 1-1: Fixed nodes, varying noise ===")
    mechanisms = {
        "INDP": INDPMechanism(),
        "Disco": DiscoMechanism(),
        "ALOHA": ALOHAMechanism(),
    }
    results = {name: [] for name in mechanisms}

    for nf in tqdm(noise_factors, desc="Noise levels"):
        for name, mech in mechanisms.items():
            accs = []
            for trial in range(n_trials):
                env = OceanEnv(noise_factor=nf, render_mode="none")
                env.reset(seed=trial * 100 + int(nf * 10))
                for _ in range(10):
                    env.step()
                gt = env.topology_mgr.adjacency
                mech.reset()
                discovered, _ = mech.discover(env.nodes, noise_factor=nf)
                acc = compute_accuracy(discovered, gt)
                accs.append(acc)
                env.close()
            results[name].append(np.mean(accs))

    if save_plots:
        plt.figure(figsize=(8, 5))
        markers = {"INDP": "o-", "Disco": "s--", "ALOHA": "^:"}
        for name, vals in results.items():
            plt.plot(noise_factors, vals, markers[name], label=name,
                     linewidth=2, markersize=7)
        plt.xlabel("Environmental Noise Factor", fontsize=12)
        plt.ylabel("Discovery Accuracy", fontsize=12)
        plt.title("Exp 1-1: Accuracy vs. Noise (Fixed Node Count)",
                  fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "exp1_1_noise_accuracy.png"),
                    dpi=200)
        plt.close()
        print("  Saved exp1_1_noise_accuracy.png")

    return noise_factors, results


def run_exp1_2(scale_factors=None, noise_factor=2.0, n_trials=5,
               save_plots=True):
    """Fixed noise, varying node count -> accuracy."""
    if scale_factors is None:
        scale_factors = [0.5, 0.75, 1.0, 1.5, 2.0]

    print("\n=== Exp 1-2: Fixed noise, varying node count ===")
    base_counts = {"satellite": 3, "uav": 6, "ship": 10,
                   "buoy": 20, "base_station": 3}
    total_nodes_list = []

    mechanisms = {
        "INDP": INDPMechanism(),
        "Disco": DiscoMechanism(),
        "ALOHA": ALOHAMechanism(),
    }
    results = {name: [] for name in mechanisms}

    for sf in tqdm(scale_factors, desc="Scale factors"):
        counts = {k: max(1, int(v * sf)) for k, v in base_counts.items()}
        total = sum(counts.values())
        total_nodes_list.append(total)
        for name, mech in mechanisms.items():
            accs = []
            for trial in range(n_trials):
                env = OceanEnv(node_counts=counts, noise_factor=noise_factor,
                               render_mode="none")
                env.reset(seed=trial * 200 + int(sf * 100))
                for _ in range(10):
                    env.step()
                gt = env.topology_mgr.adjacency
                mech.reset()
                discovered, _ = mech.discover(env.nodes,
                                              noise_factor=noise_factor)
                acc = compute_accuracy(discovered, gt)
                accs.append(acc)
                env.close()
            results[name].append(np.mean(accs))

    if save_plots:
        plt.figure(figsize=(8, 5))
        markers = {"INDP": "o-", "Disco": "s--", "ALOHA": "^:"}
        for name, vals in results.items():
            plt.plot(total_nodes_list, vals, markers[name], label=name,
                     linewidth=2, markersize=7)
        plt.xlabel("Total Number of Nodes", fontsize=12)
        plt.ylabel("Discovery Accuracy", fontsize=12)
        plt.title("Exp 1-2: Accuracy vs. Node Count (Fixed Noise)",
                  fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "exp1_2_nodes_accuracy.png"),
                    dpi=200)
        plt.close()
        print("  Saved exp1_2_nodes_accuracy.png")

    return total_nodes_list, results


def run_exp1(noise_factors=None, scale_factors=None, noise_factor=2.0,
             n_trials=5, save_xlsx=True, save_plots=True):
    """Run both Exp1-1 and Exp1-2, optionally saving xlsx."""
    nf_list, results_noise = run_exp1_1(
        noise_factors=noise_factors, n_trials=n_trials, save_plots=save_plots)
    tn_list, results_nodes = run_exp1_2(
        scale_factors=scale_factors, noise_factor=noise_factor,
        n_trials=n_trials, save_plots=save_plots)

    if save_xlsx:
        mech_names = list(results_noise.keys())
        _save_exp1_xlsx(nf_list, tn_list, results_noise, results_nodes,
                        mech_names)

    return results_noise, results_nodes


if __name__ == "__main__":
    run_exp1()
    print("\nAll Exp-1 plots saved to P1/results/")

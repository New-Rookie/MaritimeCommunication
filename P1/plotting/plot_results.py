"""
Plot generation for Research Content 1 experiments.

Produces 5 figures as specified in Experiment Manual I:
  Fig1 — F1_topo vs eta_N (mechanism comparison)
  Fig2 — F1_topo vs N_total (mechanism comparison)
  Fig3 — Reward curve vs training episode (Improved IPPO lr sweep)
  Fig4 — E_ND vs eta_N (algorithm comparison)
  Fig5 — E_ND vs N_total (algorithm comparison)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Consistent style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.figsize": (7, 5),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

MECH_MARKERS = {"INDP": "o", "Disco": "s", "ALOHA": "^"}
MECH_COLORS = {"INDP": "#2196F3", "Disco": "#FF9800", "ALOHA": "#4CAF50"}

ALGO_MARKERS = {"Improved_IPPO": "o", "IPPO": "s", "Greedy": "^",
                "ACO": "D", "GA": "v"}
ALGO_COLORS = {"Improved_IPPO": "#E91E63", "IPPO": "#2196F3",
               "Greedy": "#FF9800", "ACO": "#4CAF50", "GA": "#9C27B0"}

LR_COLORS = {1e-4: "#2196F3", 3e-4: "#FF9800", 1e-3: "#4CAF50"}
LR_LABELS = {1e-4: "lr=1e-4", 3e-4: "lr=3e-4", 1e-3: "lr=1e-3"}


def _safe_load(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — F1 vs noise (mechanism)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig1(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_a_summary.csv"))
    if df is None:
        print("  [Fig1] block_a_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for mech in ["INDP", "Disco", "ALOHA"]:
        sub = df[df["mechanism"] == mech].sort_values("eta_N")
        ax.errorbar(sub["eta_N"], sub["mean"], yerr=sub["std"],
                    marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                    label=mech, capsize=3, linewidth=1.5)
    ax.set_xlabel("Noise scale $\\eta_N$")
    ax.set_ylabel("$F1_{topo}$")
    ax.set_title("Mechanism Comparison — Accuracy vs Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig1_RC1_accuracy_vs_noise.png"))
    plt.close(fig)
    print("  [Fig1] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — F1 vs N_total (mechanism)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig2(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_b_summary.csv"))
    if df is None:
        print("  [Fig2] block_b_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for mech in ["INDP", "Disco", "ALOHA"]:
        sub = df[df["mechanism"] == mech].sort_values("N_total")
        ax.errorbar(sub["N_total"], sub["mean"], yerr=sub["std"],
                    marker=MECH_MARKERS[mech], color=MECH_COLORS[mech],
                    label=mech, capsize=3, linewidth=1.5)
    ax.set_xlabel("Total node count $N_{total}$")
    ax.set_ylabel("$F1_{topo}$")
    ax.set_title("Mechanism Comparison — Accuracy vs Node Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig2_RC1_accuracy_vs_nodes.png"))
    plt.close(fig)
    print("  [Fig2] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Reward curve (lr sweep)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig3(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_c_summary.csv"))
    if df is None:
        print("  [Fig3] block_c_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for lr_val in sorted(df["lr"].unique()):
        sub = df[df["lr"] == lr_val].sort_values("episode")
        color = LR_COLORS.get(lr_val, "gray")
        label = LR_LABELS.get(lr_val, f"lr={lr_val}")
        ax.plot(sub["episode"], sub["mean"], color=color, label=label,
                linewidth=1.5)
        ax.fill_between(sub["episode"],
                        sub["mean"] - sub["std"],
                        sub["mean"] + sub["std"],
                        alpha=0.15, color=color)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean episodic reward")
    ax.set_title("Improved IPPO — Reward Curve (Learning Rate Sweep)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig3_RC1_reward_lr.png"))
    plt.close(fig)
    print("  [Fig3] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — E_ND vs noise (algorithm)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig4(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_d_summary.csv"))
    if df is None:
        print("  [Fig4] block_d_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for algo in ["Improved_IPPO", "IPPO", "Greedy", "ACO", "GA"]:
        sub = df[df["algorithm"] == algo].sort_values("eta_N")
        if sub.empty:
            continue
        ax.errorbar(sub["eta_N"], sub["mean"], yerr=sub["std"],
                    marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                    label=algo.replace("_", " "), capsize=3, linewidth=1.5)
    ax.set_xlabel("Noise scale $\\eta_N$")
    ax.set_ylabel("Mean $E_{ND}$ (J)")
    ax.set_title("Algorithm Comparison — Energy vs Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig4_RC1_energy_vs_noise.png"))
    plt.close(fig)
    print("  [Fig4] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — E_ND vs N_total (algorithm)
# ═══════════════════════════════════════════════════════════════════════════

def plot_fig5(log_dir: str, fig_dir: str):
    df = _safe_load(os.path.join(log_dir, "block_e_summary.csv"))
    if df is None:
        print("  [Fig5] block_e_summary.csv not found, skipping.")
        return

    fig, ax = plt.subplots()
    for algo in ["Improved_IPPO", "IPPO", "Greedy", "ACO", "GA"]:
        sub = df[df["algorithm"] == algo].sort_values("N_total")
        if sub.empty:
            continue
        ax.errorbar(sub["N_total"], sub["mean"], yerr=sub["std"],
                    marker=ALGO_MARKERS[algo], color=ALGO_COLORS[algo],
                    label=algo.replace("_", " "), capsize=3, linewidth=1.5)
    ax.set_xlabel("Total node count $N_{total}$")
    ax.set_ylabel("Mean $E_{ND}$ (J)")
    ax.set_title("Algorithm Comparison — Energy vs Node Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(fig_dir, "Fig5_RC1_energy_vs_nodes.png"))
    plt.close(fig)
    print("  [Fig5] saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(log_dir: str = "P1/logs",
                         fig_dir: str = "P1/figures"):
    os.makedirs(fig_dir, exist_ok=True)
    print("\nGenerating Research Content 1 figures...")
    plot_fig1(log_dir, fig_dir)
    plot_fig2(log_dir, fig_dir)
    plot_fig3(log_dir, fig_dir)
    plot_fig4(log_dir, fig_dir)
    plot_fig5(log_dir, fig_dir)
    print("Done.\n")


if __name__ == "__main__":
    generate_all_figures()

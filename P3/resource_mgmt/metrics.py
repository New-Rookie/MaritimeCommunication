"""
Metrics module for Research Content 3 resource management experiments.

Provides aggregate statistics computed from a batch of OffloadResult objects.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .task_offloader import OffloadResult


def aggregate_results(results: List[OffloadResult], Gamma_max: float = 1e8) -> Dict[str, float]:
    """Compute per-window aggregate metrics from offloading results."""
    if not results:
        return {
            "mean_T_total": 1e6,
            "mean_E_total": 1e6,
            "mean_Gamma": 0.0,
            "success_rate": 0.0,
            "mean_alpha": 0.0,
            "throughput_normalised": 0.0,
        }

    T_vals = np.array([r.T_total for r in results], dtype=np.float64)
    E_vals = np.array([r.E_total for r in results], dtype=np.float64)
    G_vals = np.array([r.Gamma for r in results], dtype=np.float64)
    successes = np.array([r.success for r in results], dtype=np.float64)
    alphas = np.array([r.alpha_off for r in results], dtype=np.float64)

    return {
        "mean_T_total": float(np.mean(T_vals)),
        "mean_E_total": float(np.mean(E_vals)),
        "mean_Gamma": float(np.mean(G_vals)),
        "success_rate": float(np.mean(successes)),
        "mean_alpha": float(np.mean(alphas)),
        "throughput_normalised": float(np.mean(G_vals) / max(Gamma_max, 1.0)),
    }


def compute_reward(
    metrics: Dict[str, float],
    T_max: float,
    E_max: float,
    Gamma_max: float,
    lambda_G: float = 1.0,
    lambda_T: float = 1.0,
    lambda_E: float = 1.0,
    lambda_V: float = 2.0,
) -> float:
    """
    Scalar reward following Manuscript III:
      r = λ_Γ (Γ/Γ_max) − λ_T (T/T_max) − λ_E (E/E_max) − λ_V 1{violation}
    """
    norm_G = metrics["throughput_normalised"]
    norm_T = min(metrics["mean_T_total"] / max(T_max, 1e-9), 10.0)
    norm_E = min(metrics["mean_E_total"] / max(E_max, 1e-9), 10.0)
    violation = 1.0 if (metrics["mean_T_total"] > T_max or
                        metrics["mean_E_total"] > E_max) else 0.0

    return (lambda_G * norm_G
            - lambda_T * norm_T
            - lambda_E * norm_E
            - lambda_V * violation)

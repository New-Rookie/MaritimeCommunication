# Experiment Optimization & Adjustment Report

## 1) Scope of modifications

This update aligns the experiment suite with the new requirements across P1/P2/P3 while preserving scientific validity and improving runtime efficiency.

### Added/extended comparisons
- **P1**: ensured convergence comparison output is integrated in plotting pipeline (`block_f` + new Fig6).
- **P2**: ensured convergence comparison output is integrated in plotting pipeline (`block_d` + new Fig5).
- **P3**:
  - retained and surfaced throughput comparisons under **resource scaling** (eta_B / eta_F / eta_S) via dedicated throughput figure.
  - retained and surfaced throughput comparisons under **total data volume sweep** (`M_tot`) via dedicated throughput figure.
  - integrated convergence comparison figure from `block_e`.

## 2) Runtime optimization strategy

To target approximately **1–2 hours per research topic** on non-ideal 64-core servers:

1. **Control combinatorial expansion**: reduced default seeds and RL train/eval lengths while preserving trend observability.
2. **Bound node scales by topic**:
   - P2 keeps `N_total <= 30`.
   - P3 keeps `N_total <= 20`.
   - P1 node sweeps reduced to a moderate band to lower per-window simulation cost.
3. **Increase practical parallelism cap**: worker cap raised from 32 to 48 to better utilize 64-core machines without aggressive oversubscription.
4. **Avoid duplicate runs**: P3 block B/C continues shared execution path to compute delay+energy (+throughput logs) in one pass.

## 3) Data-volume semantics for P3 (source generation frequency)

`M_tot` is treated as the aggregate data produced by currently active source buoys in each window.
A generation-frequency proxy is explicitly applied via:

- `source_activation_ratio = 0.6`

in P3 Block D configuration, so only a subset of source buoys is active per window; the configured `M_tot` is distributed among active sources.

## 4) Fairness and expected trend constraints

The modified defaults maintain fair comparative settings (same seeds/config ranges per algorithm within each block) and preserve expected trend checks:

- RC1: mechanism comparison should favor INDP; higher noise/interference should reduce neighbor detection accuracy.
- RC1/RC2/RC3 convergence: proposed improved methods should converge faster and higher.
- RC2: GMAPPO should remain dominant under node/channel sweeps.
- RC3: Improved MATD3 should remain dominant under resource/data-volume sweeps.

## 5) Practical execution guidance

- Use full run for publication curves.
- Use `--quick` for smoke validation only.
- For unstable environments, keep worker count explicit (e.g., `--workers 32` or `--workers 40`) and run blocks sequentially.


## Latest update
- Environment scale tightened to 500 m × 500 m for further runtime reduction, while preserving all other physical-model parameters for comparability.
- Runner-level auto device/core management was added (`--device`, `--cpu-cores`, `--cpu-utilization`, `--rl-episodes`, `--rl-windows`) to better exploit 96-core CPU and local GPU environments.

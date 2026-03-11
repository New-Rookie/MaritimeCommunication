# 96-Core Operational Guide (P1 + P2 + P3 concurrent)

## 1. Why this now runs faster

The biggest runtime blocker was GT topology generation in `Env/core_env.py`:
- old: nested Python loops over node pairs + repeated scalar channel calls;
- new: vectorized expected-SNR matrix via `compute_gt_snr_matrix_vectorized(...)` and candidate filtering in batches.

This preserves the same GT formula and thresholds while removing millions of Python calls per campaign.

## 2. Recommended launch profiles

## A) 96-core CPU-only server (run all three concurrently)
Use explicit budgets to avoid accidental oversubscription:

```bash
python -m P1.experiments.runner --device cpu --cpu-cores 32 --workers 32 --rl-episodes 800 --rl-windows 10 --log-dir P1/logs
python -m P2.experiments.runner --device cpu --cpu-cores 32 --workers 32 --rl-episodes 800 --rl-windows 10 --log-dir P2/logs
python -m P3.experiments.runner --device cpu --cpu-cores 32 --workers 32 --rl-episodes 800 --rl-windows 10 --log-dir P3/logs
```

## B) Local machine with GPU
Single runner (or staggered runners) can use auto GPU detection:

```bash
python -m P3.experiments.runner --device auto --cpu-utilization 1.0 --rl-episodes 800 --rl-windows 10 --log-dir P3/logs
```

If CUDA is available, runner selects GPU-oriented execution mode automatically.

## 3. tmux parallel launch pattern

```bash
tmux new-session -d -s exp96 'python -m P1.experiments.runner --device cpu --cpu-cores 32 --workers 32 --rl-episodes 800 --rl-windows 10 --log-dir P1/logs'
tmux split-window -h 'python -m P2.experiments.runner --device cpu --cpu-cores 32 --workers 32 --rl-episodes 800 --rl-windows 10 --log-dir P2/logs'
tmux split-window -v 'python -m P3.experiments.runner --device cpu --cpu-cores 32 --workers 32 --rl-episodes 800 --rl-windows 10 --log-dir P3/logs'
tmux select-layout tiled
tmux attach -t exp96
```

## 4. Validity guardrails

- Keep all algorithm comparisons inside each block under identical config/seed policy.
- GT topology uses fixed `gt_eta_N` and same `T_min` contact condition as before (mathematics unchanged, only vectorized batching).
- If publication confidence intervals are needed later, increase `N_SEEDS` only for final reported runs.

## 5. Runtime troubleshooting checklist

- If progress stalls, reduce `--workers` slightly (e.g., 28 per runner) to reduce process contention.
- If GPU memory pressure appears, lower simultaneous GPU-heavy runs (run one RL-heavy topic at a time on GPU).
- Verify log growth in `P*/logs/block_*_raw.csv` every few minutes to confirm progress.

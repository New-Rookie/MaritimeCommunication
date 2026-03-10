# Experiment Reconfiguration Plan (96-core concurrent execution)

## Mandatory adjustments implemented
1. Environment area reduced to **1×1 km** via `area_width=1000 m`, `area_height=1000 m`.
2. **P1** node policy:
   - varying-node blocks capped at 80: `{20,35,50,65,80}`
   - fixed-node blocks set to `N_total=50`
3. **P2** node policy:
   - varying-node blocks capped at 35: `{10,15,20,25,30,35}`
   - fixed-node blocks set to `N_total=20`
4. **P3** all blocks fixed to `N_total=15`.
5. **All experiment blocks** now use `N_SEEDS=1`.
6. Runner auto-workers redesigned for concurrent execution:
   - each runner defaults to `cpu_count // 3` workers.
   - on a 96-core machine, each of P1/P2/P3 gets 32 workers by default.

## 96-core concurrent scheduling
When launching all three runners simultaneously, default autoshare gives:
- P1: 32 workers
- P2: 32 workers
- P3: 32 workers

This avoids oversubscription from each runner independently selecting full CPU count.

## Experimental validity analysis
- **Comparability preserved**: all algorithms inside the same block still run under identical environment, seed policy, and metric definitions.
- **Scale reduction rationale**: area scaling reduces simulation cost while maintaining the same physical/channel formulas and reward definitions.
- **Node-count policies**: fixed/varying node counts now exactly follow your requested constraints.
- **Single-seed risk**: variance confidence decreases; trend-level comparisons remain feasible but confidence intervals become less robust. If publication-grade confidence is required later, increase `N_SEEDS` only for final runs.
- **Runtime safety**: tensor-shape stabilization in MATD3 variants + `M_tot` division guards reduce risks of dimension mismatch and division-by-zero edge cases.

## Recommended launch commands
```bash
python -m P1.experiments.runner --log-dir P1/logs
python -m P2.experiments.runner --log-dir P2/logs
python -m P3.experiments.runner --log-dir P3/logs
```
(or run in three separate terminals/process managers concurrently)

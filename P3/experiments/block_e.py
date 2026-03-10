"""Block E — convergence comparison for RL algorithms (RC3)."""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from tqdm import tqdm
from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from P3.algorithms.improved_matd3 import ImprovedMATD3
from P3.algorithms.matd3 import MATD3

N_SEEDS=6; N_EPISODES=60; N_WINDOWS=5
ALGO_NAMES=['Improved_MATD3','MATD3']

def _worker(args):
    algo,seed,n_episodes,n_windows=args
    cfg=EnvConfig(N_total=20, print_diagnostics=False)
    env=MarineIoTEnv(cfg, mode='resource_mgmt', max_steps=n_windows*20+50)
    rng=np.random.default_rng(seed)
    agent = ImprovedMATD3(min(cfg.N_src, cfg.node_counts["buoy"]),cfg,lr=3e-4) if algo=='Improved_MATD3' else MATD3(min(cfg.N_src, cfg.node_counts["buoy"]),cfg,lr=3e-4)
    rows=[]
    for ep in range(n_episodes):
        info=agent.train_episode(env,n_windows=n_windows,rng=rng)
        rows.append({'experiment':'E','algorithm':algo,'seed':seed,'episode':ep,'mean_reward':info['mean_reward']})
    env.close(); return rows

def run_block_e(log_dir='P3/logs', n_seeds=N_SEEDS, n_episodes=N_EPISODES, n_windows=N_WINDOWS, n_workers=None):
    os.makedirs(log_dir,exist_ok=True)
    n_workers=min(os.cpu_count() or 1,48) if n_workers is None else n_workers
    units=[(a,s,n_episodes,n_windows) for a in ALGO_NAMES for s in range(n_seeds)]
    rows=[]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for batch in tqdm(pool.map(_worker, units), total=len(units), desc='Block E', unit='cfg', dynamic_ncols=True): rows.extend(batch)
    df=pd.DataFrame(rows); df.to_csv(os.path.join(log_dir,'block_e_raw.csv'),index=False)
    summary=df.groupby(['algorithm','episode'])['mean_reward'].agg(['mean','std']).reset_index()
    summary.to_csv(os.path.join(log_dir,'block_e_summary.csv'),index=False)
    return summary

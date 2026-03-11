"""Block D — convergence comparison for RL algorithms (RC2)."""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from tqdm import tqdm
from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from P2.link_quality.rf_estimator import LinkQualityEstimator
from P2.algorithms.gmappo import GMAPPO
from P2.algorithms.mappo import MAPPO

N_SEEDS=1; N_EPISODES=60; N_WINDOWS=10
ALGO_NAMES=['GMAPPO','MAPPO']

def _worker(args):
    algo,seed,estimator_path,n_episodes,n_windows,device=args
    cfg=EnvConfig(N_total=20, eta_ch=1.0, print_diagnostics=False)
    env=MarineIoTEnv(cfg, mode='link_selection', max_steps=n_windows*20+50)
    est=LinkQualityEstimator()
    if estimator_path and os.path.exists(estimator_path): est.load(estimator_path)
    rng=np.random.default_rng(seed)
    agent = GMAPPO(cfg.N_total,cfg,est,lr=3e-4,device=device) if algo=='GMAPPO' else MAPPO(cfg.N_total,cfg,est,lr=3e-4,device=device)
    rows=[]
    for ep in range(n_episodes):
        info=agent.train_episode(env,n_windows=n_windows,rng=rng)
        rows.append({'experiment':'D','algorithm':algo,'seed':seed,'episode':ep,'mean_reward':info['mean_reward']})
    env.close(); return rows

def run_block_d(log_dir='P2/logs', estimator_path=None, n_seeds=N_SEEDS, n_episodes=N_EPISODES, n_windows=N_WINDOWS, n_workers=None, device="cpu"):
    os.makedirs(log_dir,exist_ok=True)
    n_workers=min(os.cpu_count() or 1,48) if n_workers is None else n_workers
    units=[(a,s,estimator_path,n_episodes,n_windows,device) for a in ALGO_NAMES for s in range(n_seeds)]
    rows=[]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for batch in tqdm(pool.map(_worker, units), total=len(units), desc='Block D', unit='cfg', dynamic_ncols=True): rows.extend(batch)
    df=pd.DataFrame(rows); df.to_csv(os.path.join(log_dir,'block_d_raw.csv'),index=False)
    summary=df.groupby(['algorithm','episode'])['mean_reward'].agg(['mean','std']).reset_index()
    summary.to_csv(os.path.join(log_dir,'block_d_summary.csv'),index=False)
    return summary

"""Block F — convergence comparison for RL algorithms (RC1)."""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from tqdm import tqdm
from Env.config import EnvConfig
from Env.core_env import MarineIoTEnv
from P1.protocols.indp import INDPProtocol
from P1.algorithms.improved_ippo import ImprovedIPPO
from P1.algorithms.ippo import IPPO

N_SEEDS=1
N_EPISODES=60
N_WINDOWS=5
ALGO_NAMES=["Improved_IPPO","IPPO"]

def _worker(args):
    algo, seed, n_episodes, n_windows = args
    cfg=EnvConfig(N_total=50, eta_N=1.0, print_diagnostics=False)
    env=MarineIoTEnv(cfg, mode='discovery', max_steps=n_windows*cfg.N_slot)
    protocol=INDPProtocol(cfg)
    rng=np.random.default_rng(seed)
    agent = ImprovedIPPO(cfg.N_total, cfg=cfg, lr=3e-4) if algo=="Improved_IPPO" else IPPO(cfg.N_total, cfg=cfg, lr=3e-4)
    rows=[]
    for ep in range(n_episodes):
        info=agent.train_episode(env, protocol, n_windows=n_windows, rng=rng)
        rows.append({"experiment":"F","algorithm":algo,"seed":seed,"episode":ep,"mean_reward":info["mean_reward"]})
    env.close(); return rows

def run_block_f(log_dir='P1/logs', n_seeds=N_SEEDS, n_episodes=N_EPISODES, n_windows=N_WINDOWS, n_workers=None):
    os.makedirs(log_dir,exist_ok=True)
    n_workers = min(os.cpu_count() or 1, 48) if n_workers is None else n_workers
    units=[(a,s,n_episodes,n_windows) for a in ALGO_NAMES for s in range(n_seeds)]
    rows=[]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for batch in tqdm(pool.map(_worker, units), total=len(units), desc='Block F', unit='cfg', dynamic_ncols=True):
            rows.extend(batch)
    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(log_dir,'block_f_raw.csv'),index=False)
    summary=df.groupby(['algorithm','episode'])['mean_reward'].agg(['mean','std']).reset_index()
    summary.to_csv(os.path.join(log_dir,'block_f_summary.csv'),index=False)
    return summary

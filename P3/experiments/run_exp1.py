"""
P3 – Experiment 1: Varying resource conditions (fixed noise & interference).

9 sub-experiments:
  Rows: vary bandwidth / compute / storage (while fixing the other two)
  Cols: total delay / total energy / throughput

Compare: MADDPG, MATD3, Greedy, GA, IMATD3

All algorithms train in parallel via multiprocessing.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from P3.rm_env import ResourceManagementEnv, HISTORY_LEN
from P3.models.imatd3 import IMATD3Agent
from P3.models.matd3 import MATD3Agent
from P3.models.maddpg import MADDPG_RM_Agent
from P3.models.greedy import GreedyRM
from P3.models.genetic import GeneticRM

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_EPISODES = 600
EVAL_EPISODES = 15
EP_LEN = 200
NOISE_FIXED = 2.0
UPDATE_EVERY = 4

ALGO_NAMES = ["MADDPG", "MATD3", "Greedy", "GA", "IMATD3"]
STYLES = {"MADDPG": "D--", "MATD3": "s-.", "Greedy": "v:",
          "GA": "x--", "IMATD3": "o-"}


# ===================================================================
# Train + Evaluate (standalone, each creates its own agent)
# ===================================================================

def _train_eval(env, algo_name, train_ep, eval_ep, ep_len):
    obs_dim = env.obs_dim
    act_dim = env.action_dim

    if algo_name == "IMATD3":
        agent = IMATD3Agent(obs_dim=obs_dim, action_dim=act_dim,
                            global_state_dim=64)
    elif algo_name == "MATD3":
        agent = MATD3Agent(obs_dim=obs_dim, action_dim=act_dim, state_dim=64)
    elif algo_name == "MADDPG":
        agent = MADDPG_RM_Agent(obs_dim=obs_dim, action_dim=act_dim,
                                state_dim=64)
    elif algo_name == "Greedy":
        agent = GreedyRM()
    elif algo_name == "GA":
        agent = GeneticRM(action_dim=act_dim)
    else:
        raise ValueError(algo_name)

    for ep in range(train_ep):
        obs_dict, _ = env.reset()
        mec_ids = list(obs_dict.keys())
        ep_reward = 0.0
        prev_obs = {mid: obs_dict[mid] for mid in mec_ids}
        prev_state = env.get_global_state()

        for step in range(ep_len):
            actions = {}
            for mid in mec_ids:
                o = prev_obs.get(mid, np.zeros(obs_dim))
                if algo_name == "IMATD3":
                    a = agent.select_action(o, agent_id=mid, explore=True)
                elif algo_name in ("MATD3", "MADDPG"):
                    a = agent.select_action(o, explore=True)
                elif algo_name == "GA":
                    a = agent.select_action(o)
                else:
                    a = agent.select_action(o)
                actions[mid] = a

            if algo_name == "IMATD3":
                pre_state_hist = env.get_state_history()

            obs_dict, r, done, _, info = env.step(actions)
            ep_reward += r
            cur_state = env.get_global_state()

            if algo_name == "IMATD3":
                post_state_hist = env.get_state_history()

            for mid in mec_ids:
                o_new = obs_dict.get(mid, np.zeros(obs_dim))
                if algo_name == "IMATD3":
                    oh = env.get_obs_history(
                        mid, prev_obs.get(mid, np.zeros(obs_dim)))
                    noh = env.get_obs_history(mid, o_new)
                    agent.store(oh, actions[mid], r, noh, done,
                                pre_state_hist, post_state_hist)
                elif algo_name in ("MATD3", "MADDPG"):
                    agent.store(prev_obs.get(mid, np.zeros(obs_dim)),
                                actions[mid], r, o_new, done,
                                prev_state, cur_state)
                prev_obs[mid] = o_new
            prev_state = cur_state

            if algo_name in ("IMATD3", "MATD3", "MADDPG"):
                if step % UPDATE_EVERY == 0:
                    agent.update()
            if done:
                break

        if algo_name == "IMATD3":
            agent.end_episode(ep_reward)
        if algo_name == "GA":
            agent.set_fitness(ep_reward)
            if (ep + 1) % agent.pop_size == 0:
                agent.evolve()

    delays, energies, throughputs = [], [], []
    for _ in range(eval_ep):
        obs_dict, _ = env.reset()
        mec_ids = list(obs_dict.keys())
        for step in range(ep_len):
            actions = {}
            for mid in mec_ids:
                o = obs_dict.get(mid, np.zeros(obs_dim))
                if algo_name == "IMATD3":
                    a = agent.select_action(o, agent_id=mid, explore=False)
                elif algo_name in ("MATD3", "MADDPG"):
                    a = agent.select_action(o, explore=False)
                else:
                    a = agent.select_action(o)
                actions[mid] = a
            obs_dict, _, done, _, info = env.step(actions)
            if done:
                break
        delays.append(info["total_delay"])
        energies.append(info["total_energy"])
        throughputs.append(info["total_throughput"])

    return {
        "delay": float(np.mean(delays)),
        "energy": float(np.mean(energies)),
        "throughput": float(np.mean(throughputs)),
    }


# ===================================================================
# Parallel workers (module-level)
# ===================================================================

def _init_pool_worker(gpu_frac):
    import matplotlib as _mpl
    _mpl.use("Agg")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if gpu_frac > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_frac)


def _pool_worker_exp1(args):
    """Standalone worker for P3 experiments."""
    algo, env_kwargs, train_ep, eval_ep, ep_len = args
    env = ResourceManagementEnv(**env_kwargs)
    return _train_eval(env, algo, train_ep, eval_ep, ep_len)


# ===================================================================
# xlsx saving
# ===================================================================

def _save_exp1_xlsx(resource_results, algo_names):
    try:
        from openpyxl import Workbook
    except ImportError:
        print("  [WARN] openpyxl not installed, skipping xlsx save")
        return

    for res_name, (scales, label, results) in resource_results.items():
        for metric in ["delay", "energy", "throughput"]:
            wb = Workbook()
            ws = wb.active
            ws.title = metric
            ws.append([label] + algo_names)
            for i, s in enumerate(scales):
                row = [s]
                for algo in algo_names:
                    row.append(results[algo][metric][i])
                ws.append(row)
            fname = f"exp1_{res_name}_{metric}_data.xlsx"
            wb.save(os.path.join(RESULTS_DIR, fname))
            print(f"  Saved {fname}")


# ===================================================================
# Main experiment function
# ===================================================================

def run_exp1(train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES,
             ep_len=EP_LEN, noise_factor=NOISE_FIXED,
             max_workers=5, gpu_frac=0.15,
             save_xlsx=True, save_plots=True):
    print("\n=== P3 Exp-1: Varying resource conditions ===")

    resource_configs = {
        "bandwidth": {
            "scales": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            "label": "Bandwidth Scale Factor",
            "make_kwargs": lambda s: dict(
                noise_factor=noise_factor, episode_length=ep_len,
                bw_scale=s, comp_scale=1.0, stor_scale=1.0),
        },
        "compute": {
            "scales": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            "label": "Compute Scale Factor",
            "make_kwargs": lambda s: dict(
                noise_factor=noise_factor, episode_length=ep_len,
                bw_scale=1.0, comp_scale=s, stor_scale=1.0),
        },
        "storage": {
            "scales": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            "label": "Storage Scale Factor",
            "make_kwargs": lambda s: dict(
                noise_factor=noise_factor, episode_length=ep_len,
                bw_scale=1.0, comp_scale=1.0, stor_scale=s),
        },
    }

    metrics_list = ["delay", "energy", "throughput"]
    metric_labels = {
        "delay": "Total Delay (s)",
        "energy": "Total Energy (J)",
        "throughput": "Avg Throughput (bytes/s)",
    }

    resource_results = {}
    fig_idx = 1

    for res_name, cfg in resource_configs.items():
        results = {a: {m: [] for m in metrics_list} for a in ALGO_NAMES}

        tasks = []
        for scale in cfg["scales"]:
            env_kwargs = cfg["make_kwargs"](scale)
            for algo in ALGO_NAMES:
                tasks.append((algo, env_kwargs, train_episodes,
                              eval_episodes, ep_len))

        print(f"  {res_name}: {len(tasks)} tasks, {max_workers} workers")

        if max_workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=max_workers, mp_context=ctx,
                initializer=_init_pool_worker, initargs=(gpu_frac,)
            ) as pool:
                fmap = {}
                for i, t in enumerate(tasks):
                    fmap[pool.submit(_pool_worker_exp1, t)] = i
                task_results = [None] * len(tasks)
                for f in tqdm(as_completed(fmap), total=len(tasks),
                              desc=f"Exp1-{res_name}"):
                    task_results[fmap[f]] = f.result()
        else:
            task_results = []
            for t in tqdm(tasks, desc=f"Exp1-{res_name}"):
                task_results.append(_pool_worker_exp1(t))

        idx = 0
        for _ in cfg["scales"]:
            for algo in ALGO_NAMES:
                r = task_results[idx]
                for m in metrics_list:
                    results[algo][m].append(r[m])
                idx += 1

        resource_results[res_name] = (cfg["scales"], cfg["label"], results)

        if save_plots:
            for metric in metrics_list:
                plt.figure(figsize=(8, 5))
                for algo in ALGO_NAMES:
                    plt.plot(cfg["scales"], results[algo][metric],
                             STYLES[algo], label=algo, linewidth=2,
                             markersize=7)
                plt.xlabel(cfg["label"], fontsize=12)
                plt.ylabel(metric_labels[metric], fontsize=12)
                plt.title(
                    f"P3 Exp-1-{fig_idx}: {metric_labels[metric]} "
                    f"vs. {res_name.title()}", fontsize=13)
                plt.legend(fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fname = f"exp1_{fig_idx}_{res_name}_{metric}.png"
                plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=200)
                plt.close()
                print(f"  Saved {fname}")
                fig_idx += 1

    if save_xlsx:
        _save_exp1_xlsx(resource_results, ALGO_NAMES)

    return resource_results


if __name__ == "__main__":
    run_exp1()

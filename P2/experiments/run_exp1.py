"""
P2 – Experiment 1: Different network topology conditions (fixed channel).

Compare: MADDPG, MAPPO, Greedy, ACO, MAPPO+GCN
Metrics: (1) Link switching stability  (2) Communication delay  (3) Energy

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

from P2.ls_env import LinkSelectionEnv
from P2.models.mappo_gcn import MAPPOGCNAgent
from P2.models.mappo import MAPPOAgent
from P2.models.maddpg import MADDPGAgent
from P2.models.greedy import GreedyLinkSelector
from P2.models.ant_colony import ACOLinkSelector

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_EPISODES = 800
EVAL_EPISODES = 20
EP_LEN = 200
NOISE_FIXED = 2.0
MADDPG_UPDATE_EVERY = 4

TOPO_CONFIGS = [
    {"satellite": 2, "uav": 4, "ship": 6, "buoy": 12, "base_station": 2},
    {"satellite": 3, "uav": 6, "ship": 8, "buoy": 16, "base_station": 3},
    {"satellite": 3, "uav": 6, "ship": 10, "buoy": 20, "base_station": 3},
    {"satellite": 4, "uav": 8, "ship": 12, "buoy": 25, "base_station": 4},
    {"satellite": 5, "uav": 10, "ship": 15, "buoy": 30, "base_station": 5},
]
TOPO_LABELS = [str(sum(c.values())) for c in TOPO_CONFIGS]

ALGO_NAMES = ["MADDPG", "MAPPO", "Greedy", "ACO", "MAPPO+GCN"]
STYLES = {"MADDPG": "D--", "MAPPO": "s-.", "Greedy": "v:",
          "ACO": "x--", "MAPPO+GCN": "o-"}


# ===================================================================
# Train + Evaluate functions (each creates its own agent)
# ===================================================================

def _train_and_eval_mappo_gcn(env, n_episodes, eval_ep, ep_len):
    agent = MAPPOGCNAgent(local_obs_dim=8, global_state_dim=64)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        sources = [n for n in env.ocean.nodes if n.node_type == "buoy"]
        id_to_idx = {n.id: i for i, n in enumerate(env.ocean.nodes)}

        for _ in range(ep_len):
            embeddings = agent.get_embeddings(
                obs["node_features"], obs["edge_index"])

            actions = {}
            step_data = []
            for src in sources:
                src_idx = id_to_idx.get(src.id, 0)
                emb = embeddings[src_idx].detach()
                lo = env.get_local_obs(src.id)
                mask = env.get_valid_mask(src.id)
                a, lp, _ = agent.select_action(emb, lo, mask)
                gs = env.get_global_state()
                v = agent.get_value(gs)
                actions[src.id] = a
                step_data.append({
                    "emb": emb.cpu().numpy(), "lo": lo, "mask": mask,
                    "a": a, "lp": lp, "v": v,
                })

            obs, r, done, _, info = env.step(actions)

            for sd in step_data:
                agent.store(
                    {"embedding": sd["emb"], "local_obs": sd["lo"],
                     "mask": sd["mask"]},
                    sd["a"], r, sd["lp"], sd["v"], done)

            if done:
                break
        agent.update()

    metrics = {"delay": [], "energy": [], "stability": []}
    for _ in range(eval_ep):
        obs, _ = env.reset()
        sources = [n for n in env.ocean.nodes if n.node_type == "buoy"]
        id_to_idx = {n.id: i for i, n in enumerate(env.ocean.nodes)}
        for _ in range(ep_len):
            embeddings = agent.get_embeddings(
                obs["node_features"], obs["edge_index"])
            actions = {}
            for src in sources:
                src_idx = id_to_idx.get(src.id, 0)
                emb = embeddings[src_idx].detach()
                lo = env.get_local_obs(src.id)
                mask = env.get_valid_mask(src.id)
                a, _, _ = agent.select_action(emb, lo, mask)
                actions[src.id] = a
            obs, _, done, _, info = env.step(actions)
            if done:
                break
        metrics["delay"].append(info["total_delay"])
        metrics["energy"].append(info["total_energy"])
        metrics["stability"].append(info["stability"])
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def _train_and_eval_mappo(env, n_episodes, eval_ep, ep_len):
    agent = MAPPOAgent(obs_dim=15, max_actions=15, state_dim=64)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        sources = [n for n in env.ocean.nodes if n.node_type == "buoy"]
        for _ in range(ep_len):
            actions = {}
            step_data = []
            for src in sources:
                lo = env.get_local_obs(src.id)
                padded = np.zeros(15, dtype=np.float32)
                padded[:len(lo)] = lo
                mask = env.get_valid_mask(src.id)
                a, lp = agent.select_action(padded, mask)
                v = agent.get_value(env.get_global_state())
                actions[src.id] = a
                step_data.append({"obs": padded, "a": a, "lp": lp, "v": v})

            obs, r, done, _, info = env.step(actions)

            for sd in step_data:
                agent.store(sd["obs"], sd["a"], r, sd["lp"], sd["v"], done)

            if done:
                break
        agent.update()

    metrics = {"delay": [], "energy": [], "stability": []}
    for _ in range(eval_ep):
        obs, _ = env.reset()
        sources = [n for n in env.ocean.nodes if n.node_type == "buoy"]
        for _ in range(ep_len):
            actions = {}
            for src in sources:
                lo = env.get_local_obs(src.id)
                padded = np.zeros(15, dtype=np.float32)
                padded[:len(lo)] = lo
                mask = env.get_valid_mask(src.id)
                a, _ = agent.select_action(padded, mask)
                actions[src.id] = a
            obs, _, done, _, info = env.step(actions)
            if done:
                break
        metrics["delay"].append(info["total_delay"])
        metrics["energy"].append(info["total_energy"])
        metrics["stability"].append(info["stability"])
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def _train_and_eval_maddpg(env, n_episodes, eval_ep, ep_len):
    agent = MADDPGAgent(obs_dim=15, n_actions=15, state_dim=64)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        sources = [n for n in env.ocean.nodes if n.node_type == "buoy"]
        prev_obs_map = {}
        prev_state = env.get_global_state()
        for src in sources:
            lo = env.get_local_obs(src.id)
            padded = np.zeros(15, dtype=np.float32)
            padded[:len(lo)] = lo
            prev_obs_map[src.id] = padded

        for step_i in range(ep_len):
            actions = {}
            for src in sources:
                mask = env.get_valid_mask(src.id)
                a = agent.select_action(
                    prev_obs_map.get(src.id, np.zeros(15)), mask)
                actions[src.id] = a
            obs, r, done, _, info = env.step(actions)
            cur_state = env.get_global_state()
            for src in sources:
                lo = env.get_local_obs(src.id)
                padded = np.zeros(15, dtype=np.float32)
                padded[:len(lo)] = lo
                agent.store(prev_obs_map.get(src.id, np.zeros(15)),
                            actions.get(src.id, 0), r, padded, done,
                            prev_state, cur_state)
                prev_obs_map[src.id] = padded
            prev_state = cur_state
            if step_i % MADDPG_UPDATE_EVERY == 0:
                agent.update()
            if done:
                break

    metrics = {"delay": [], "energy": [], "stability": []}
    for _ in range(eval_ep):
        obs, _ = env.reset()
        sources = [n for n in env.ocean.nodes if n.node_type == "buoy"]
        for _ in range(ep_len):
            actions = {}
            for src in sources:
                lo = env.get_local_obs(src.id)
                padded = np.zeros(15, dtype=np.float32)
                padded[:len(lo)] = lo
                mask = env.get_valid_mask(src.id)
                a = agent.select_action(padded, mask, explore=False)
                actions[src.id] = a
            _, _, done, _, info = env.step(actions)
            if done:
                break
        metrics["delay"].append(info["total_delay"])
        metrics["energy"].append(info["total_energy"])
        metrics["stability"].append(info["stability"])
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def _eval_heuristic(env, algo_cls, eval_ep, ep_len, is_aco=False):
    metrics = {"delay": [], "energy": [], "stability": []}
    for _ in range(eval_ep):
        algo = algo_cls()
        obs, _ = env.reset()
        sources = [n for n in env.ocean.nodes if n.node_type == "buoy"]
        for _ in range(ep_len):
            actions = {}
            for src in sources:
                mask = env.get_valid_mask(src.id)
                if mask.sum() < 0.5:
                    actions[src.id] = 0
                    continue
                sinr = env.get_sinr_values(src.id)
                a = algo.select_action(sinr, mask)
                actions[src.id] = a
                if is_aco:
                    algo.update(a, 1.0)
            _, _, done, _, info = env.step(actions)
            if done:
                break
        metrics["delay"].append(info["total_delay"])
        metrics["energy"].append(info["total_energy"])
        metrics["stability"].append(info["stability"])
    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ===================================================================
# Parallel worker (module-level for multiprocessing spawn)
# ===================================================================

def _init_pool_worker(gpu_frac):
    """Initializer for each pool worker process."""
    import matplotlib as _mpl
    _mpl.use("Agg")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    if gpu_frac > 0 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(gpu_frac)


def _pool_worker_exp1(args):
    """Standalone worker: creates env, trains algo, evaluates, returns metrics."""
    algo, node_counts, noise_factor, train_ep, eval_ep, ep_len = args
    env = LinkSelectionEnv(node_counts=node_counts, noise_factor=noise_factor,
                           episode_length=ep_len)
    if algo == "MAPPO+GCN":
        return _train_and_eval_mappo_gcn(env, train_ep, eval_ep, ep_len)
    elif algo == "MAPPO":
        return _train_and_eval_mappo(env, train_ep, eval_ep, ep_len)
    elif algo == "MADDPG":
        return _train_and_eval_maddpg(env, train_ep, eval_ep, ep_len)
    elif algo == "Greedy":
        return _eval_heuristic(env, GreedyLinkSelector, eval_ep, ep_len)
    elif algo == "ACO":
        return _eval_heuristic(env, ACOLinkSelector, eval_ep, ep_len, is_aco=True)
    return {"delay": 0.0, "energy": 0.0, "stability": 0.0}


# ===================================================================
# xlsx saving
# ===================================================================

def _save_exp1_xlsx(all_results, algo_names, topo_labels):
    try:
        from openpyxl import Workbook
    except ImportError:
        print("  [WARN] openpyxl not installed, skipping xlsx save")
        return

    for metric in ["stability", "delay", "energy"]:
        wb = Workbook()
        ws = wb.active
        ws.title = metric
        ws.append(["Total_Nodes"] + algo_names)
        for i, label in enumerate(topo_labels):
            row = [label]
            for algo in algo_names:
                row.append(all_results[algo][metric][i])
            ws.append(row)
        fname = f"exp1_{metric}_data.xlsx"
        wb.save(os.path.join(RESULTS_DIR, fname))
        print(f"  Saved {fname}")


# ===================================================================
# Main experiment function
# ===================================================================

def run_exp1(train_episodes=TRAIN_EPISODES, eval_episodes=EVAL_EPISODES,
             ep_len=EP_LEN, noise_factor=NOISE_FIXED,
             topo_configs=None, max_workers=5, gpu_frac=0.15,
             save_xlsx=True, save_plots=True):
    if topo_configs is None:
        topo_configs = TOPO_CONFIGS
    topo_labels = [str(sum(c.values())) for c in topo_configs]
    algo_names = ALGO_NAMES

    all_results = {a: {"delay": [], "energy": [], "stability": []}
                   for a in algo_names}

    tasks = []
    for counts in topo_configs:
        for algo in algo_names:
            tasks.append((algo, counts, noise_factor,
                          train_episodes, eval_episodes, ep_len))

    print(f"\n=== P2 Exp-1: Varying topology ({len(tasks)} tasks, "
          f"{max_workers} workers) ===")

    if max_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx,
                                 initializer=_init_pool_worker,
                                 initargs=(gpu_frac,)) as pool:
            future_map = {}
            for i, t in enumerate(tasks):
                future_map[pool.submit(_pool_worker_exp1, t)] = i
            results = [None] * len(tasks)
            for f in tqdm(as_completed(future_map), total=len(tasks),
                          desc="P2-Exp1"):
                results[future_map[f]] = f.result()
    else:
        results = []
        for t in tqdm(tasks, desc="P2-Exp1"):
            results.append(_pool_worker_exp1(t))

    idx = 0
    for _ in topo_configs:
        for algo in algo_names:
            m = results[idx]
            for k in ["delay", "energy", "stability"]:
                all_results[algo][k].append(m[k])
            idx += 1

    if save_xlsx:
        _save_exp1_xlsx(all_results, algo_names, topo_labels)

    if save_plots:
        metric_labels = {
            "stability": ("Link Switching Stability", "exp1_1_stability.png"),
            "delay": ("Communication Delay (s)", "exp1_2_delay.png"),
            "energy": ("Communication Energy (J)", "exp1_3_energy.png"),
        }
        for metric, (ylabel, fname) in metric_labels.items():
            plt.figure(figsize=(8, 5))
            for algo in algo_names:
                plt.plot(topo_labels, all_results[algo][metric],
                         STYLES[algo], label=algo, linewidth=2, markersize=7)
            plt.xlabel("Total Number of Nodes", fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.title(f"P2 Exp-1: {ylabel} vs. Topology (Fixed Channel)",
                      fontsize=13)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=200)
            plt.close()
            print(f"  Saved {fname}")

    return all_results


if __name__ == "__main__":
    run_exp1()

"""
P2 实验入口：基于GMAPPO的海洋异构网络链路选择策略研究

所有可调参数集中在下方配置区，直接修改即可。
运行方式: python run_p2.py
"""

import sys
import os
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
os.environ["PYTHONPATH"] = _ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["SDL_VIDEODRIVER"] = "dummy"

# ====================================================================
#                         可调参数配置区
# ====================================================================

# --- 训练参数 ---
TRAIN_EPISODES = 800           # RL算法训练轮数
EVAL_EPISODES = 20             # 评估轮数
EPISODE_LENGTH = 200           # 每轮步数

# --- Exp 1: 不同拓扑条件 (固定信道) ---
EXP1_NOISE_FACTOR = 2.0       # 固定噪声因子
EXP1_TOPO_CONFIGS = [
    {"satellite": 2, "uav": 4, "ship": 6, "buoy": 12, "base_station": 2},
    {"satellite": 3, "uav": 6, "ship": 8, "buoy": 16, "base_station": 3},
    {"satellite": 3, "uav": 6, "ship": 10, "buoy": 20, "base_station": 3},
    {"satellite": 4, "uav": 8, "ship": 12, "buoy": 25, "base_station": 4},
    {"satellite": 5, "uav": 10, "ship": 15, "buoy": 30, "base_station": 5},
]

# --- Exp 2: 不同信道条件 (固定拓扑) ---
EXP2_NODE_COUNTS = {"satellite": 3, "uav": 6, "ship": 10,
                    "buoy": 20, "base_station": 3}
EXP2_NOISE_LEVELS = [0.5, 1.0, 2.0, 4.0, 7.0, 10.0]

# --- 并行 & GPU ---
MAX_WORKERS = 5                # 并行进程数 (设为1则串行)
GPU_MEMORY_FRACTION = 0.15     # 每进程GPU显存占比

# --- 输出 ---
SAVE_XLSX = True               # 保存xlsx数据文件
SAVE_PLOTS = True              # 保存图片

# ====================================================================


def main():
    import matplotlib
    matplotlib.use("Agg")

    print("=" * 60)
    print("  P2: 链路选择实验 (Link Selection)")
    print("=" * 60)
    t0 = time.time()

    from P2.experiments.run_exp1 import run_exp1
    from P2.experiments.run_exp2 import run_exp2

    print("\n--- Experiment 1: 不同拓扑条件 ---")
    run_exp1(
        train_episodes=TRAIN_EPISODES,
        eval_episodes=EVAL_EPISODES,
        ep_len=EPISODE_LENGTH,
        noise_factor=EXP1_NOISE_FACTOR,
        topo_configs=EXP1_TOPO_CONFIGS,
        max_workers=MAX_WORKERS,
        gpu_frac=GPU_MEMORY_FRACTION,
        save_xlsx=SAVE_XLSX,
        save_plots=SAVE_PLOTS,
    )

    print("\n--- Experiment 2: 不同信道条件 ---")
    run_exp2(
        train_episodes=TRAIN_EPISODES,
        eval_episodes=EVAL_EPISODES,
        ep_len=EPISODE_LENGTH,
        node_counts=EXP2_NODE_COUNTS,
        noise_levels=EXP2_NOISE_LEVELS,
        max_workers=MAX_WORKERS,
        gpu_frac=GPU_MEMORY_FRACTION,
        save_xlsx=SAVE_XLSX,
        save_plots=SAVE_PLOTS,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  P2 全部实验完成  耗时: {elapsed/60:.1f} min")
    print(f"  结果保存在: P2/results/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""
P3 实验入口：基于IMATD3的海洋异构网络多维资源管理研究

所有可调参数集中在下方配置区，直接修改即可。
运行方式: python run_p3.py
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
TRAIN_EPISODES = 600           # RL算法训练轮数
EVAL_EPISODES = 15             # 评估轮数
EPISODE_LENGTH = 200           # 每轮步数

# --- Exp 1: 不同资源条件 (固定噪声) ---
EXP1_NOISE_FACTOR = 2.0       # 固定噪声因子

# --- Exp 2: 不同噪声条件 (固定资源) ---
EXP2_NOISE_LEVELS = [0.5, 1.0, 2.0, 4.0, 7.0, 10.0]

# --- Exp 3: 不同节点数量 (固定噪声 & 资源) ---
EXP3_NOISE_FACTOR = 2.0
EXP3_NODE_CONFIGS = [
    {"satellite": 2, "uav": 3, "ship": 5, "buoy": 10, "base_station": 2},
    {"satellite": 3, "uav": 5, "ship": 8, "buoy": 15, "base_station": 3},
    {"satellite": 3, "uav": 6, "ship": 10, "buoy": 20, "base_station": 3},
    {"satellite": 4, "uav": 8, "ship": 13, "buoy": 25, "base_station": 4},
    {"satellite": 5, "uav": 10, "ship": 16, "buoy": 30, "base_station": 5},
]

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
    print("  P3: 资源管理实验 (Resource Management)")
    print("=" * 60)
    t0 = time.time()

    from P3.experiments.run_exp1 import run_exp1
    from P3.experiments.run_exp2 import run_exp2
    from P3.experiments.run_exp3 import run_exp3

    print("\n--- Experiment 1: 不同资源条件 ---")
    run_exp1(
        train_episodes=TRAIN_EPISODES,
        eval_episodes=EVAL_EPISODES,
        ep_len=EPISODE_LENGTH,
        noise_factor=EXP1_NOISE_FACTOR,
        max_workers=MAX_WORKERS,
        gpu_frac=GPU_MEMORY_FRACTION,
        save_xlsx=SAVE_XLSX,
        save_plots=SAVE_PLOTS,
    )

    print("\n--- Experiment 2: 不同噪声条件 ---")
    run_exp2(
        train_episodes=TRAIN_EPISODES,
        eval_episodes=EVAL_EPISODES,
        ep_len=EPISODE_LENGTH,
        noise_levels=EXP2_NOISE_LEVELS,
        max_workers=MAX_WORKERS,
        gpu_frac=GPU_MEMORY_FRACTION,
        save_xlsx=SAVE_XLSX,
        save_plots=SAVE_PLOTS,
    )

    print("\n--- Experiment 3: 不同节点数量 ---")
    run_exp3(
        train_episodes=TRAIN_EPISODES,
        eval_episodes=EVAL_EPISODES,
        ep_len=EPISODE_LENGTH,
        noise_factor=EXP3_NOISE_FACTOR,
        node_configs=EXP3_NODE_CONFIGS,
        max_workers=MAX_WORKERS,
        gpu_frac=GPU_MEMORY_FRACTION,
        save_xlsx=SAVE_XLSX,
        save_plots=SAVE_PLOTS,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  P3 全部实验完成  耗时: {elapsed/60:.1f} min")
    print(f"  结果保存在: P3/results/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

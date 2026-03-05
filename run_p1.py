"""
P1 实验入口：基于IPPO的海洋异构网络邻居发现研究

所有可调参数集中在下方配置区，直接修改即可。
运行方式: python run_p1.py
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

# --- Exp 1: 机制对比 (INDP / Disco / ALOHA) ---
EXP1_NOISE_FACTORS = [0.0, 1.0, 2.0, 4.0, 7.0, 10.0]
EXP1_SCALE_FACTORS = [0.5, 0.75, 1.0, 1.5, 2.0]
EXP1_NOISE_FACTOR = 2.0        # Exp1-2 使用的固定噪声
EXP1_N_TRIALS = 5              # 每个配置点的重复次数

# --- Exp 2: 算法对比 (PPO / IPPO / ACO / Greedy / GA) ---
EXP2_TRAIN_EPISODES = 2500     # 训练轮数
EXP2_EPISODE_LENGTH = 50       # 每轮步数
EXP2_NOISE_FACTOR = 2.0        # 训练默认噪声
EXP2_NOISE_FACTORS = [0.0, 1.0, 2.0, 4.0, 7.0, 10.0]
EXP2_SCALE_FACTORS = [0.5, 0.75, 1.0, 1.5, 2.0]
EXP2_EVAL_TRIALS = 3           # 评估重复次数

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
    print("  P1: 邻居发现实验 (Neighbour Discovery)")
    print("=" * 60)
    t0 = time.time()

    from P1.experiments.run_exp1 import run_exp1
    from P1.experiments.run_exp2 import run_exp2_1, run_exp2_2_to_5

    print("\n--- Experiment 1: 机制对比 (INDP / Disco / ALOHA) ---")
    run_exp1(
        noise_factors=EXP1_NOISE_FACTORS,
        scale_factors=EXP1_SCALE_FACTORS,
        noise_factor=EXP1_NOISE_FACTOR,
        n_trials=EXP1_N_TRIALS,
        save_xlsx=SAVE_XLSX,
        save_plots=SAVE_PLOTS,
    )

    print("\n--- Experiment 2-1: 算法奖励曲线 ---")
    run_exp2_1(
        episodes=EXP2_TRAIN_EPISODES,
        ep_len=EXP2_EPISODE_LENGTH,
        noise_factor=EXP2_NOISE_FACTOR,
        max_workers=MAX_WORKERS,
        gpu_frac=GPU_MEMORY_FRACTION,
        save_xlsx=SAVE_XLSX,
        save_plots=SAVE_PLOTS,
    )

    print("\n--- Experiment 2-2~2-5: 算法评估 ---")
    run_exp2_2_to_5(
        episodes=EXP2_TRAIN_EPISODES,
        ep_len=EXP2_EPISODE_LENGTH,
        noise_factors=EXP2_NOISE_FACTORS,
        scale_factors=EXP2_SCALE_FACTORS,
        train_noise=EXP2_NOISE_FACTOR,
        n_trials=EXP2_EVAL_TRIALS,
        max_workers=MAX_WORKERS,
        gpu_frac=GPU_MEMORY_FRACTION,
        save_xlsx=SAVE_XLSX,
        save_plots=SAVE_PLOTS,
    )

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  P1 全部实验完成  耗时: {elapsed/60:.1f} min")
    print(f"  结果保存在: P1/results/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

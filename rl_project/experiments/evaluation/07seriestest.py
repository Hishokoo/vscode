# 07seriestest.py
# 07 系列模型的評估工具（純評估，不進行任何訓練）
#
# 兩種用法：
#   1. 視覺化播放 —— 開啟 MuJoCo 視窗，直接看模型在模擬器裡怎麼控制單擺
#   2. 量化評估   —— 無視窗跑大量 episode，輸出可比較的能力數據
#
# 使用範例（一律在 rl_project/ 目錄下執行）：
#   python experiments/evaluation/07seriestest.py --render
#       預設情境播放 3 集，開視窗
#   python experiments/evaluation/07seriestest.py --render --scenario worst --episodes 5
#       最壞情境播放 5 集
#   python experiments/evaluation/07seriestest.py
#       四種情境各跑 30 集，輸出完整能力報告（無視窗，較快）
#   python experiments/evaluation/07seriestest.py --scenario realistic --episodes 100
#       單一情境跑 100 集，統計更穩定
#   python experiments/evaluation/07seriestest.py --model output/models/05114
#       改測其他模型

import argparse
import importlib
import os
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3

# ==========================================
# 路徑設定：本檔位於 experiments/evaluation/，往上兩層回到 rl_project/
# ==========================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_MACHINE_DIR = os.path.join(_RL_PROJECT_ROOT, "experiments", "machine")

# AdvancedMotorWrapper 定義在 07111（位於 experiments/machine/），需加入 sys.path 才能 import
if _MACHINE_DIR not in sys.path:
    sys.path.insert(0, _MACHINE_DIR)

_mod = importlib.import_module("07111trytrain_mujoco_td3")
AdvancedMotorWrapper = _mod.AdvancedMotorWrapper


# ==========================================
# 情境預設
# ==========================================
# 每個情境代表一種「硬體條件」假設。比較同一個模型在不同情境下的表現，
# 就能知道它的能力邊界落在哪裡，而不只是一個籠統的分數。
_SENSOR = (0.001, 0.001, 0.02, 0.02)   # per-channel 感測雜訊，四種情境共用

SCENARIOS = {
    # 理想狀況：只有感測雜訊，沒有任何硬體缺陷。用來確認模型的能力上限。
    "clean": dict(
        delay_steps=0, alpha=0.9, obs_noise_std=_SENSOR,
    ),

    # 真實預期：編碼器直連 MCU 的有線系統。延遲 2~10 ms 不到一步，掉幀率低。
    # 這是最接近「實際上機會遇到什麼」的情境。
    "realistic": dict(
        delay_steps=0, obs_noise_std=_SENSOR,
        alpha_randomize=True, alpha_range=(0.85, 0.98),
        angle_bias_enable=True, angle_bias_range=(-0.01, 0.01),
        friction_randomize=True, friction_slide_range=(0.0, 4.0),
        friction_hinge_range=(0.0, 0.01),
        gain_error_enable=True, gain_error_range=(0.85, 1.15),
        jitter_enable=True, jitter_prob=0.02,
        disturbance_enable=True, disturbance_probability=0.3,
        disturbance_start_range=(50, 300),
        disturbance_magnitude_range=(0.1, 0.3),
        disturbance_duration_range=(2, 5),
    ),

    # 訓練情境：與 07111 訓練時完全相同的設定（延遲隨機 0/1）。
    # 用來確認模型在自己的訓練分布內表現如何。
    "full": dict(
        obs_noise_std=_SENSOR,
        delay_randomize=True, delay_random_range=(0, 1),
        alpha_randomize=True, alpha_range=(0.85, 0.98),
        angle_bias_enable=True, angle_bias_range=(-0.01, 0.01),
        friction_randomize=True, friction_slide_range=(0.0, 4.0),
        friction_hinge_range=(0.0, 0.01),
        gain_error_enable=True, gain_error_range=(0.85, 1.15),
        jitter_enable=True, jitter_prob=0.02,
        disturbance_enable=True, disturbance_probability=0.3,
        disturbance_start_range=(50, 300),
        disturbance_magnitude_range=(0.1, 0.3),
        disturbance_duration_range=(2, 5),
    ),

    # 最壞情況：延遲固定 1 步、掉幀率 5%、摩擦固定在上限、外力擾動必定發生。
    # 用來找安全邊際 —— 這個情境撐得住，實體上就有餘裕。
    "worst": dict(
        delay_steps=1, alpha=0.85, obs_noise_std=_SENSOR,
        angle_bias_enable=True, angle_bias_range=(-0.01, 0.01),
        friction_randomize=True, friction_slide_range=(3.5, 4.0),
        friction_hinge_range=(0.008, 0.01),
        gain_error_enable=True, gain_error_range=(0.85, 1.15),
        jitter_enable=True, jitter_prob=0.05,
        disturbance_enable=True, disturbance_probability=1.0,
        disturbance_start_range=(50, 300),
        disturbance_magnitude_range=(0.2, 0.3),
        disturbance_duration_range=(3, 5),
    ),
}

MAX_STEPS = 2000   # 每集上限（= 80 秒模擬時間）


def make_env(scenario, render=False):
    """建立評估環境。render=True 會開啟 MuJoCo 視窗並以 25 fps 即時播放。"""
    base = gym.make(
        "InvertedPendulum-v4",
        max_episode_steps=MAX_STEPS,
        render_mode="human" if render else None,
    )
    return AdvancedMotorWrapper(base, **SCENARIOS[scenario])


def run_episode(model, env):
    """
    跑完一集，回傳該集的量化指標。
    所有指標都用物理引擎的「乾淨」狀態計算，不含感測雜訊與零點偏移 ——
    我們要衡量的是桿子真實的狀態，不是模型以為的狀態。
    """
    obs, _ = env.reset()
    angles, positions, actions = [], [], []
    steps = 0

    for _ in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1

        # 直接從物理引擎讀真實狀態（obs 已被雜訊與 bias 污染，不能用）
        angles.append(float(env.unwrapped.data.qpos[1]))
        positions.append(float(env.unwrapped.data.qpos[0]))
        actions.append(float(action[0]))

        if terminated or truncated:
            break

    a = np.array(angles)
    p = np.array(positions)
    u = np.array(actions)
    n = len(a)

    return {
        "steps": steps,
        "success": steps >= MAX_STEPS,
        "ise_per_step": float(np.sum(a ** 2) / n),
        "sse": float(np.mean(np.abs(a[int(n * 0.8):]))) if n >= 5 else float("nan"),
        "overshoot": float(np.max(np.abs(a))),
        "rms_effort": float(np.sqrt(np.mean(u ** 2))),
        "pos_ise_per_step": float(np.sum(p ** 2) / n),
        "max_cart_pos": float(np.max(np.abs(p))),
        "rail_hit": bool(env._rail_limit_hit),
    }


def summarize(results, label):
    """把多集的結果彙整成一行摘要。"""
    steps = np.array([r["steps"] for r in results])
    succ = np.mean([r["success"] for r in results]) * 100
    rail = np.mean([r["rail_hit"] for r in results]) * 100
    sse = np.array([r["sse"] for r in results], dtype=float)

    def m(key):
        return float(np.mean([r[key] for r in results]))

    print(f"\n{'=' * 64}")
    print(f"情境：{label}    共 {len(results)} 集")
    print("-" * 64)
    print(f"  滿分率（撐滿 {MAX_STEPS} 步）   {succ:>8.1f} %      ← 最重要")
    print(f"  存活步數  中位 {np.median(steps):>6.0f}   平均 {steps.mean():>6.0f}   最短 {steps.min():>6.0f}")
    print(f"  撞邊界終止比例                {rail:>8.1f} %      ← 應為 0")
    print("-" * 64)
    print(f"  角度誤差 ISE/步               {m('ise_per_step'):>10.5f}   越小越好")
    print(f"  穩態誤差 SSE                  {np.nanmean(sse):>10.5f}   rad，越小越好")
    print(f"  最大超調                      {m('overshoot'):>10.5f}   rad（終止門檻 0.2）")
    print(f"  位置誤差 ISE/步               {m('pos_ise_per_step'):>10.5f}   越小越好")
    print(f"  滑車最遠位置                  {m('max_cart_pos'):>10.5f}   m（邊界 0.9）")
    print(f"  控制力 RMS                    {m('rms_effort'):>10.5f}   滿格 3.0，對應馬達發熱")


def main():
    ap = argparse.ArgumentParser(
        description="07 系列模型評估：MuJoCo 視覺化播放 + 量化能力指標"
    )
    ap.add_argument("-m", "--model", default="output/models/07111",
                    help="模型路徑（不含 .zip），預設 output/models/07111")
    ap.add_argument("-s", "--scenario", default=None, choices=list(SCENARIOS),
                    help="評估情境；不指定則四種全跑")
    ap.add_argument("-n", "--episodes", type=int, default=None,
                    help="每個情境跑幾集（--render 預設 3，量化模式預設 30）")
    ap.add_argument("-r", "--render", action="store_true",
                    help="開啟 MuJoCo 視窗即時播放（25 fps）")
    args = ap.parse_args()

    model_path = os.path.join(_RL_PROJECT_ROOT, args.model)
    if not os.path.exists(model_path + ".zip"):
        print(f"❌ 找不到模型：{model_path}.zip")
        sys.exit(1)

    model = TD3.load(model_path)
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS)
    n_ep = args.episodes if args.episodes else (3 if args.render else 30)

    print(f"模型：{args.model}.zip")
    print(f"模式：{'視覺化播放（MuJoCo 視窗）' if args.render else '量化評估（無視窗）'}")

    for sc in scenarios:
        env = make_env(sc, render=args.render)
        results = []
        for ep in range(n_ep):
            r = run_episode(model, env)
            results.append(r)
            if args.render:
                mark = "✅ 撐滿" if r["success"] else ("🧱 撞邊界" if r["rail_hit"] else "❌ 倒下")
                print(f"  [{sc}] 第 {ep + 1} 集：{r['steps']:>5} 步  {mark}   "
                      f"最大角度 {r['overshoot']:.4f}  滑車最遠 {r['max_cart_pos']:.3f} m")
        env.close()
        summarize(results, sc)

    print(f"\n{'=' * 64}")
    if not args.render:
        print("提示：加上 --render 可開啟 MuJoCo 視窗直接觀察控制行為")


if __name__ == "__main__":
    main()

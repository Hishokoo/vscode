# 07worst_ablation.py
# 找出 worst 情境滿分率掉到 53.3% 的關鍵參數（純評估，不訓練）
#
# 兩個方向的消融，所有變體共用同一組 seed（common random numbers）：
#   A. 由壞往好（leave-one-out）：從 worst 出發，一次只把「一組」參數還原成訓練值
#      → 還原哪一組後滿分率回升最多，那組就是主因
#   B. 由好往壞（add-one-in）：從「full + delay 固定 1」出發，一次只加入一組 worst 的改動
#      → 加入哪一組後掉最多，那組就是主因
# 另外對每一集失敗做歸因：開局即死 / 擾動後倒下 / 抖動後倒下 / 其他
#
# 放在 experiments/evaluation/ 下，於 rl_project/ 目錄執行：
#   python experiments/evaluation/07worst_ablation.py              # 每變體 100 集
#   python experiments/evaluation/07worst_ablation.py -n 200       # 更穩定，較慢
#   python experiments/evaluation/07worst_ablation.py --only A      # 只跑方向 A

import argparse
import importlib
import os
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

_t = importlib.import_module("07seriestest")      # 沿用同一份情境定義與 wrapper
SCENARIOS, AdvancedMotorWrapper, MAX_STEPS = _t.SCENARIOS, _t.AdvancedMotorWrapper, _t.MAX_STEPS

# ==========================================
# 參數組：worst 版本 vs 訓練（full）版本
# ==========================================
GROUPS = {
    "jitter": dict(
        worst=dict(jitter_prob=0.05),
        train=dict(jitter_prob=0.02),
    ),
    "disturbance": dict(
        worst=dict(disturbance_probability=1.0, disturbance_magnitude_range=(0.2, 0.3),
                   disturbance_duration_range=(3, 5)),
        train=dict(disturbance_probability=0.3, disturbance_magnitude_range=(0.1, 0.3),
                   disturbance_duration_range=(2, 5)),
    ),
    "friction": dict(
        worst=dict(friction_slide_range=(3.5, 4.0), friction_hinge_range=(0.008, 0.01)),
        train=dict(friction_slide_range=(0.0, 4.0), friction_hinge_range=(0.0, 0.01)),
    ),
    "alpha": dict(
        worst=dict(alpha_randomize=False, alpha=0.85),
        train=dict(alpha_randomize=True, alpha_range=(0.85, 0.98)),
    ),
    "delay": dict(
        worst=dict(delay_randomize=False, delay_steps=1),
        train=dict(delay_randomize=True, delay_random_range=(0, 1)),
    ),
}

BASE_A = dict(SCENARIOS["worst"])
BASE_B = dict(SCENARIOS["full"], delay_randomize=False, delay_steps=1)   # 訓練分布中 delay=1 的那一半


def variants(which):
    out = []
    if which in ("A", "both"):
        out.append(("A | worst（原樣）", BASE_A))
        for g, v in GROUPS.items():
            out.append((f"A | worst，還原 {g}", dict(BASE_A, **v["train"])))
    if which in ("B", "both"):
        out.append(("B | full+delay=1（基準）", BASE_B))
        for g, v in GROUPS.items():
            if g != "delay":                              # 基準已固定 delay=1
                out.append((f"B | 基準，加入 worst {g}", dict(BASE_B, **v["worst"])))
        all_worst = dict(BASE_B)
        for g, v in GROUPS.items():
            all_worst.update(v["worst"])
        out.append(("B | 基準，全部加入（應≈worst）", all_worst))
    return out


# ==========================================
# 單集執行 + 失敗歸因
# ==========================================
def run_episode(model, env, seed):
    np.random.seed(seed)                                  # wrapper 用的是全域 np.random
    obs, _ = env.reset(seed=seed)                          # MuJoCo 自帶 reset 雜訊
    theta0 = float(env.unwrapped.data.qpos[1])             # 含 wrapper 的 ±0.15 初始擾動

    dist_start = None
    jitter_steps = []
    prev_applied = env._last_applied_action.copy()

    for k in range(1, MAX_STEPS + 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)

        if dist_start is None and env._disturbance_steps_remaining > 0:
            dist_start = k
        # 抖動觸發時 applied 會完全等於上一步（浮點數巧合相等機率可忽略）
        if k > 1 and np.array_equal(env._last_applied_action, prev_applied):
            jitter_steps.append(k)
        prev_applied = env._last_applied_action.copy()

        if terminated or truncated:
            break

    success = k >= MAX_STEPS
    cause = None
    if not success:
        recent_jitter = [j for j in jitter_steps if k - 5 <= j <= k]
        if k < 50:
            cause = "開局即死"                              # 擾動最早第 50 步才開始
        elif dist_start is not None and 0 <= k - dist_start <= 25:
            cause = "擾動後倒下"
        elif recent_jitter:
            cause = "抖動後倒下"
        else:
            cause = "其他"
    return dict(steps=k, success=success, cause=cause, theta0=abs(theta0),
                n_jitter=len(jitter_steps))


def wilson(p, n, z=1.96):
    """滿分率的 95% 信賴區間（Wilson），用來判斷差異是否大於抽樣雜訊。"""
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="output/models/07111")
    ap.add_argument("-n", "--episodes", type=int, default=100)
    ap.add_argument("--only", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    model = TD3.load(os.path.join(_RL_PROJECT_ROOT, args.model))
    seeds = list(range(10_000, 10_000 + args.episodes))
    causes = ["開局即死", "擾動後倒下", "抖動後倒下", "其他"]

    print(f"模型 {args.model}.zip，每變體 {args.episodes} 集，共用 seed {seeds[0]}~{seeds[-1]}")
    print(f"\n{'變體':<34}{'滿分率':>8}  {'95% CI':>15}  " + "  ".join(f"{c:>6}" for c in causes))
    print("-" * 110)

    for label, cfg in variants(args.only):
        env = AdvancedMotorWrapper(gym.make("InvertedPendulum-v4", max_episode_steps=MAX_STEPS), **cfg)
        res = [run_episode(model, env, s) for s in seeds]
        env.close()

        p = np.mean([r["success"] for r in res])
        lo, hi = wilson(p, len(res))
        cnt = [sum(r["cause"] == c for r in res) for c in causes]
        print(f"{label:<34}{p * 100:>7.1f}%  [{lo * 100:5.1f}, {hi * 100:5.1f}]  "
              + "  ".join(f"{c:>6}" for c in cnt))

        early = [r["theta0"] for r in res if r["cause"] == "開局即死"]
        if early:
            print(f"{'':<34}  開局即死的 |θ0|：平均 {np.mean(early):.3f}，最小 {np.min(early):.3f} rad")

    print("\n讀法：A 組看「還原哪一組，滿分率回升最多」；B 組看「加入哪一組，滿分率掉最多」。")
    print("      兩組 CI 不重疊才算確定的差異；只有組合後才掉的話，代表是交互作用而非單一參數。")


if __name__ == "__main__":
    main()

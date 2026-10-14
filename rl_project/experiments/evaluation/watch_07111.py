"""
watch_07111.py — 載入 TD3 訓練好的 .zip，觀看 / 錄製 InvertedPendulum-v4 模擬動畫

環境會包上與 07111 訓練時相同的 AdvancedMotorWrapper（觀測 5 維），
情境設定沿用 07seriestest.SCENARIOS，不另外複製一份。

用法範例（一律在 rl_project/ 目錄下執行）：
  # 即時視窗觀看 3 個 episode（預設情境 full = 與訓練時完全相同的領域隨機化）
  python experiments/evaluation/watch_07111.py --episodes 3

  # 換情境：clean / realistic / full / worst
  python experiments/evaluation/watch_07111.py --scenario clean

  # 只關掉隨機外力擾動，摩擦/增益誤差/零點偏移/jitter 保留
  python experiments/evaluation/watch_07111.py --scenario realistic --no-disturbance

  # 錄成 mp4（需要 pip install "imageio[ffmpeg]"），預設存到 output/07111_demo.mp4
  python experiments/evaluation/watch_07111.py --mode video

  # 擾動測試：第 300 步給擺桿一個角速度衝擊
  python experiments/evaluation/watch_07111.py --push-step 300 --push-vel 1.5

  # 週期推力：每 4 秒推一次，大小 U(0, 0.5) rad/s、方向隨機（會關閉情境內的隨機外力）
  python experiments/evaluation/watch_07111.py --push-every 4 --push-max 0.5 --max-steps 500

  # 改看其他 07 系列模型
  python experiments/evaluation/watch_07111.py --model output/models/07111
"""
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
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 07seriestest 已負責把 experiments/machine/ 加入 sys.path 並載入 07111 的 wrapper
_t = importlib.import_module("07seriestest")
AdvancedMotorWrapper = _t.AdvancedMotorWrapper
SCENARIOS = _t.SCENARIOS


def make_env(render_mode, max_steps, scenario, no_disturbance=False):
    base = gym.make("InvertedPendulum-v4", render_mode=render_mode,
                    max_episode_steps=max_steps)
    cfg = dict(SCENARIOS[scenario])
    if no_disturbance:
        cfg["disturbance_enable"] = False   # 只關隨機外力，摩擦/增益/bias/jitter 照情境保留
    return AdvancedMotorWrapper(base, **cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="output/models/07111",
                   help="相對於 rl_project/ 的模型路徑")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="full",
                   help="full = 與 07111 訓練時相同的設定")
    p.add_argument("--no-disturbance", action="store_true",
                   help="關閉情境內的隨機外力擾動，其他領域隨機化保留")
    p.add_argument("--mode", choices=["human", "video"], default="human")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=2000, help="與訓練時相同為 2000")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--push-step", type=int, default=-1)
    p.add_argument("--push-vel", type=float, default=0.0, help="施加於擺桿角速度的增量 rad/s")
    p.add_argument("--push-every", type=float, default=0.0,
                   help="每隔幾秒推一次擺桿（>0 啟用；大小 U(0, --push-max)、方向隨機；會關閉情境內的隨機外力）")
    p.add_argument("--push-max", type=float, default=0.5, help="週期推力的最大角速度增量 rad/s")
    p.add_argument("--out", default="output/07111_demo.mp4", help="相對於 rl_project/")
    p.add_argument("--plot", action="store_true", help="結束後畫出角度與動作曲線")
    args = p.parse_args()

    # mp4 需要 ffmpeg 後端；先檢查，避免跑完整段模擬才在存檔時失敗（.gif 不需要）
    if args.mode == "video" and args.out.endswith(".mp4"):
        try:
            import imageio_ffmpeg  # noqa: F401
        except ImportError:
            sys.exit('存 mp4 需要 ffmpeg：pip install "imageio[ffmpeg]"，或改用 --out output/07111_demo.gif')

    # wrapper 內部的隨機化都用 np.random 全域 RNG，env.reset(seed=...) 管不到，要另外設
    np.random.seed(args.seed)

    render_mode = "human" if args.mode == "human" else "rgb_array"
    env = make_env(render_mode, args.max_steps, args.scenario,
                   args.no_disturbance or args.push_every > 0)
    ctrl_dt = env.unwrapped.dt                         # InvertedPendulum-v4：0.04 s
    push_period = int(round(args.push_every / ctrl_dt)) if args.push_every > 0 else 0
    model = TD3.load(os.path.join(_RL_PROJECT_ROOT, args.model), device="cpu")

    frames, logs = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_ret, t, angles, actions = 0.0, 0, [], []
        push_rng = np.random.default_rng(args.seed + ep)      # 週期推力序列，同 seed 可重播
        print(f"--- Episode {ep + 1} / {args.episodes}（seed {args.seed + ep}）---")
        while True:
            action, _ = model.predict(obs, deterministic=True)

            # 擾動：直接修改 MuJoCo 狀態中擺桿的角速度 (qvel[1])
            if t == args.push_step:
                env.unwrapped.data.qvel[1] += args.push_vel
                print(f"[ep {ep}] step {t}: push Δω = {args.push_vel} rad/s")
            if push_period and t > 0 and t % push_period == 0:
                dv = push_rng.uniform(0.0, args.push_max) * push_rng.choice([-1.0, 1.0])
                env.unwrapped.data.qvel[1] += dv
                print(f"   [ep {ep}] t={t * ctrl_dt:.1f}s 推擺桿 Δθ̇ = {dv:+.2f} rad/s")

            obs, r, term, trunc, info = env.step(action)
            ep_ret += r
            t += 1
            # obs[1] 已含 angle_bias 與感測雜訊，畫圖用物理引擎的真實角度
            angles.append(float(env.unwrapped.data.qpos[1]))
            actions.append(float(action[0]))

            if args.mode == "video":
                frames.append(env.render())
            if term or trunc:
                break

        m = info.get("control_metrics", {})
        if trunc and not term:
            status = "撐到上限 (truncated)"
        elif m.get("rail_limit_hit", 0.0) > 0:
            status = "滑車超出 rail_limit (terminated)"
        else:
            status = "倒下 (terminated)"
        print(f"Episode {ep}: steps={t}, return={ep_ret:.1f}, {status}, "
              f"max|θ|={np.max(np.abs(angles)):.4f} rad, "
              f"delay={m.get('dr_delay_steps', float('nan')):.0f}, "
              f"friction={m.get('dr_friction_slide', float('nan')):.2f} N")
        logs.append((np.array(angles), np.array(actions)))

    env.close()

    if args.mode == "video":
        import imageio
        out_path = os.path.join(_RL_PROJECT_ROOT, args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # 模擬步長 dt=0.04 s → 25 fps 才是真實速度
        fps = env.metadata.get("render_fps", 25)
        imageio.mimsave(out_path, frames, fps=fps)
        print(f"影片已存到 {out_path}（{len(frames)} 幀, {fps} fps）")

    if args.plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 5))
        for i, (a, u) in enumerate(logs):
            ax[0].plot(a, label=f"ep {i}")
            ax[1].plot(u, label=f"ep {i}")
        ax[0].set_ylabel("pole angle θ (rad)")
        ax[1].set_ylabel("action (policy output)")
        ax[1].set_xlabel("step")
        if args.push_step >= 0:
            for a_ in ax:
                a_.axvline(args.push_step, ls="--", c="gray")
        ax[0].legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

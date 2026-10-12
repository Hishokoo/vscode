"""
watch_07113.py — 載入 07113 匯出的 actor.pt，在 IP570 MuJoCo 模型上觀看 / 錄製動畫

用的是上機時同一條推論路徑：actor.pt（只含 actor.mu）+ 6 維觀測（ObservationBuilder），
不需要 SB3、也不需要 critic，所以 asym critic 的模型（E5 / E6）也直接可用。
環境、控制頻率與 07113 訓練時相同（頻率從同名的 _config.json 讀取）。

用法（在 rl_project/ 目錄下執行）：
  # 即時視窗（macOS 的 MuJoCo 視窗必須用 mjpython 啟動）
  mjpython experiments/evaluation/watch_07113.py

  # 全開 DR（物理 / 感測 / 通訊隨機化 + 隨機外力），預設是名目模型
  mjpython experiments/evaluation/watch_07113.py --dr

  # 擾動測試：第 5 秒給擺桿角速度 +1.0 rad/s
  mjpython experiments/evaluation/watch_07113.py --push-sec 5 --push-vel 1.0

  # 錄成 gif（不需要視窗；mp4 需要 pip install "imageio[ffmpeg]"）
  python experiments/evaluation/watch_07113.py --mode video --out output/07113_E5_demo.gif

  # 換模型
  mjpython experiments/evaluation/watch_07113.py --model output/models/07113_E2_25hz_s0_actor.pt
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch as th
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # rl_project/
TRAIN_SCRIPT = PROJECT_ROOT / "experiments" / "machine" / "07113trytrain_mujoco_td3.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("ip570_07113", TRAIN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_actor(path):
    """依 actor.pt 記錄的 net_arch 重建 actor.mu：Linear-ReLU-…-Linear-Tanh。"""
    d = th.load(path, map_location="cpu", weights_only=False)
    dims = [d["obs_dim"], *d["net_arch"], d["act_dim"]]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU() if i < len(dims) - 2 else nn.Tanh())
    mu = nn.Sequential(*layers)
    mu.load_state_dict(d["state_dict"])
    mu.eval()
    return mu


def set_camera(cam):
    cam.lookat[:] = [0.0, 0.0, 0.15]
    cam.distance = 0.85
    cam.azimuth = 90.0
    cam.elevation = -8.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="output/models/07113_E5_25hz_s1_actor.pt", help="相對於 rl_project/")
    p.add_argument("--dr", action="store_true", help="全開 DR + 隨機外力（預設：名目模型、無外力）")
    p.add_argument("--mode", choices=["human", "video"], default="human")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=20_000, help="預設與最終評估同一組種子")
    p.add_argument("--push-sec", type=float, default=-1.0, help="第幾秒給擺桿角速度衝擊（<0 = 不推）")
    p.add_argument("--push-vel", type=float, default=0.0, help="擺桿角速度增量 rad/s")
    p.add_argument("--speed", type=float, default=1.0, help="播放速度倍率（human 模式）")
    p.add_argument("--out", default="output/07113_demo.gif", help="video 模式輸出，相對於 rl_project/")
    args = p.parse_args()

    model_path = PROJECT_ROOT / args.model
    config_path = model_path.with_name(model_path.name.replace("_actor.pt", "_config.json"))
    hz = json.loads(config_path.read_text(encoding="utf-8"))["timing"]["ctrl_hz"]

    m = load_train_module()
    timing = m.Timing.from_hz(hz)
    env = m.IP570Env(timing, plant_dr=args.dr, sensor_dr=args.dr, comm_dr=args.dr, disturbance=args.dr)
    mu = load_actor(model_path)
    push_step = int(round(args.push_sec / timing.ctrl_dt)) if args.push_sec >= 0 else -1
    print(f"模型 {model_path.name}｜{hz} Hz（每步 {timing.ctrl_dt * 1000:.0f} ms）｜"
          f"{'全開 DR' if args.dr else '名目模型'}｜每集最多 {timing.max_episode_steps} 步")

    viewer, renderer, frames = None, None, []
    if args.mode == "human":
        viewer = mujoco.viewer.launch_passive(env.model, env.data)
        set_camera(viewer.cam)
    else:
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        cam = mujoco.MjvCamera()
        set_camera(cam)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ret, t, done = 0.0, 0, False
        while not done:
            tic = time.perf_counter()
            with th.no_grad():
                action = mu(th.as_tensor(obs[None], dtype=th.float32)).numpy()[0]
            if t == push_step:
                env.data.qvel[1] += args.push_vel
                print(f"   [ep {ep}] t={t * timing.ctrl_dt:.1f}s 推擺桿 Δθ̇ = {args.push_vel} rad/s")
            obs, r, term, trunc, info = env.step(action)
            ret += r
            t += 1
            done = term or trunc

            if viewer is not None:
                if not viewer.is_running():
                    return
                viewer.sync()
                time.sleep(max(0.0, timing.ctrl_dt / args.speed - (time.perf_counter() - tic)))
            else:
                renderer.update_scene(env.data, camera=cam)
                frames.append(renderer.render())

        cm = info.get("control_metrics", {})
        status = ("跑滿 20 秒" if trunc and not term
                  else "撞滑軌軟限" if cm.get("rail_limit_hit") else "倒下")
        print(f"Episode {ep}: {status}，{t} 步，return={ret:.0f}，"
              f"x_std={cm.get('osc_x_std_tail', float('nan')) * 100:.2f} cm，"
              f"θ_std={cm.get('osc_theta_std_tail', float('nan')) * 1000:.1f} mrad，"
              f"max|x|={cm.get('max_cart_pos_abs', float('nan')) * 100:.1f} cm，"
              f"RMS duty={cm.get('RMS_Control_Effort', float('nan')):.3f}")

    if viewer is not None:
        print("播放結束，關閉視窗即可離開")
        while viewer.is_running():
            time.sleep(0.1)
    else:
        import imageio
        out = PROJECT_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".gif":
            imageio.mimsave(out, frames, duration=timing.ctrl_dt * 1000, loop=0)
        else:
            imageio.mimsave(out, frames, fps=hz)
        print(f"已存 {out}（{len(frames)} 幀，{hz} fps = 真實速度）")


if __name__ == "__main__":
    main()

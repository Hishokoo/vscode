"""
watch_07113.py — 載入 07113 匯出的 actor.pt，在 IP570 MuJoCo 模型上觀看 / 錄製動畫

用的是上機時同一條推論路徑：actor.pt（只含 actor.mu）+ 6 維觀測（ObservationBuilder），
不需要 SB3、也不需要 critic，所以 asym critic 的模型（E5 / E6）也直接可用。
環境、控制頻率與 07113 訓練時相同（頻率從同名的 _config.json 讀取）。

用法（在 rl_project/ 目錄下執行）：
  # 即時視窗。macOS 的 MuJoCo 視窗必須在 mjpython 底下跑；用 python 或 VS Code「執行」啟動時，
  # 本檔會自動改用同一個環境裡的 mjpython 重新啟動，所以兩種寫法都可以
  python experiments/evaluation/watch_07113.py
  mjpython experiments/evaluation/watch_07113.py

  # 預設：全開 DR（物理參數、感測（零點 / 比例 / 雜訊 / 編碼器校正）、通訊（延遲 / 掉幀）隨機化，
  # 每一集抽一組新參數，抽到的值會印出來）+ 週期推力：每 4 秒推一次擺桿，大小 U(0, 0.7) rad/s、方向隨機。
  # 週期推力開著時，環境自帶的隨機外力會關閉，推力只來自這裡；推力序列由 seed 決定，同 seed 可重播
  # 只看名目模型（DR 與所有推力全部關閉）：
  mjpython experiments/evaluation/watch_07113.py --nominal

  # 改回 07113 訓練時的隨機外力（關掉週期推力）
  mjpython experiments/evaluation/watch_07113.py --push-every 0

  # 擾動測試：第 5 秒給擺桿角速度 +1.0 rad/s
  mjpython experiments/evaluation/watch_07113.py --push-sec 5 --push-vel 1.0

  # 調整週期推力：每 2 秒推一次，上限 1.0 rad/s
  mjpython experiments/evaluation/watch_07113.py --push-every 2 --push-max 1.0

  # 錄成 gif（不需要視窗；mp4 需要 pip install "imageio[ffmpeg]"）
  python experiments/evaluation/watch_07113.py --mode video --out output/07113_E5_demo.gif

  # 換模型
  mjpython experiments/evaluation/watch_07113.py --model output/models/07113_E2_25hz_s0_actor.pt
"""
import argparse
import importlib.util
import json
import os
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


def ensure_mjpython():
    """macOS 上即時視窗需要 mjpython：不是的話，用同一個 conda 環境的 mjpython 重新執行本檔（參數不變）。"""
    if sys.platform != "darwin" or isinstance(mujoco.viewer._MJPYTHON, mujoco.viewer._MjPythonBase):
        return
    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists() or os.environ.get("WATCH_07113_REEXEC"):
        sys.exit(f"macOS 的 MuJoCo 視窗需要 mjpython，請改用：mjpython {' '.join(sys.argv)}\n"
                 "或加 --mode video 錄成 gif")
    print(f"↪ 改用 {mjpython} 重新啟動（macOS 的 MuJoCo 視窗需要 mjpython）")
    os.environ["WATCH_07113_REEXEC"] = "1"
    os.execv(str(mjpython), [str(mjpython), str(Path(__file__).resolve()), *sys.argv[1:]])


def main():
    sys.stdout.reconfigure(line_buffering=True)    # 背景 / 管線執行時也能即時看到輸出
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="output/models/07113_E5_25hz_s1_actor.pt", help="相對於 rl_project/")
    p.add_argument("--nominal", action="store_true",
                   help="關閉所有 DR、隨機外力與週期推力（預設：全開 DR + 週期推力）")
    p.add_argument("--mode", choices=["human", "video"], default="human")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=20_000, help="預設與最終評估同一組種子")
    p.add_argument("--push-sec", type=float, default=-1.0, help="第幾秒給擺桿角速度衝擊（<0 = 不推）")
    p.add_argument("--push-vel", type=float, default=0.0, help="擺桿角速度增量 rad/s")
    p.add_argument("--push-every", type=float, default=4.0,
                   help="每隔幾秒推一次擺桿（大小 U(0, --push-max)、方向隨機；會關閉環境自帶的隨機外力；0 = 不推）")
    p.add_argument("--push-max", type=float, default=0.7, help="週期推力的最大角速度增量 rad/s")
    p.add_argument("--speed", type=float, default=1.0, help="播放速度倍率（human 模式）")
    p.add_argument("--out", default="output/07113_demo.gif", help="video 模式輸出，相對於 rl_project/")
    args = p.parse_args()
    if args.mode == "human":
        ensure_mjpython()

    model_path = PROJECT_ROOT / args.model
    config_path = model_path.with_name(model_path.name.replace("_actor.pt", "_config.json"))
    hz = json.loads(config_path.read_text(encoding="utf-8"))["timing"]["ctrl_hz"]

    m = load_train_module()
    timing = m.Timing.from_hz(hz)
    dr = not args.nominal
    periodic = dr and args.push_every > 0
    env = m.IP570Env(timing, plant_dr=dr, sensor_dr=dr, comm_dr=dr, disturbance=dr and not periodic)
    mu = load_actor(model_path)
    push_step = int(round(args.push_sec / timing.ctrl_dt)) if args.push_sec >= 0 else -1
    push_period = int(round(args.push_every / timing.ctrl_dt)) if periodic else 0
    dr_text = ("全開 DR（物理 / 感測 / 通訊" + ("）" if periodic else " + 隨機外力）")) if dr else "名目模型"
    print(f"模型 {model_path.name}｜{hz} Hz（每步 {timing.ctrl_dt * 1000:.0f} ms）｜{dr_text}｜每集最多 {timing.max_episode_steps} 步"
          + (f"｜每 {args.push_every:g} 秒推一次，|Δθ̇| ≤ {args.push_max:g} rad/s" if periodic else ""))

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
        push_rng = np.random.default_rng(args.seed + ep)      # 週期推力序列，同 seed 可重播
        last_push = "—"
        if dr:
            pl, q, se, dp = env.p, env.q, env.sens, env.dist_plan
            print(f"Episode {ep} 抽到：K_F={q['K_F']:.1f} N/duty  擺 {q['m_pole'] * 1000:.0f} g  "
                  f"滑車摩擦 {pl['Fc_cart']:.1f} N  樞軸摩擦 {pl['hinge_fric'] * 1e3:.2f} mN·m  "
                  f"角度零點偏移 {se['angle_bias'] * 1e3:+.1f} mrad  延遲 {env._lat_range[0] * 2}–"
                  f"{env._lat_range[1] * 2} ms  外力 "
                  + (f"t={dp['start'] * timing.ctrl_dt:.1f}s Δθ̇={dp['dtheta_dot']:.2f} rad/s" if dp else "無"))
        while not done:
            tic = time.perf_counter()
            with th.no_grad():
                action = mu(th.as_tensor(obs[None], dtype=th.float32)).numpy()[0]
            if t == push_step:
                env.data.qvel[1] += args.push_vel
                last_push = f"{t * timing.ctrl_dt:.1f}s  {args.push_vel:+.2f} rad/s"
                print(f"   [ep {ep}] t={t * timing.ctrl_dt:.1f}s 推擺桿 Δθ̇ = {args.push_vel} rad/s")
            if push_period and t > 0 and t % push_period == 0:
                dv = push_rng.uniform(0.0, args.push_max) * push_rng.choice([-1.0, 1.0])
                env.data.qvel[1] += dv
                last_push = f"{t * timing.ctrl_dt:.1f}s  {dv:+.2f} rad/s"
                print(f"   [ep {ep}] t={t * timing.ctrl_dt:.1f}s 推擺桿 Δθ̇ = {dv:+.2f} rad/s")
            obs, r, term, trunc, info = env.step(action)
            ret += r
            t += 1
            done = term or trunc

            if viewer is not None:
                if not viewer.is_running():
                    return
                viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                  "Episode\nseed\ntime\ncart x\nlast push",
                                  f"{ep + 1} / {args.episodes}\n{args.seed + ep}\n{t * timing.ctrl_dt:5.1f} s\n"
                                  f"{env.data.qpos[0] * 100:+.1f} cm\n{last_push}"))
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

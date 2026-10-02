# 05121stress_test.py
# 05114 模型壓力測試執行檔（純評估，不進行任何訓練）
# 測試項目：馬達延遲掃描、初始推力干擾、系統質量浮動、抗擾動測試（時長 & 力大小）

import importlib
import sys
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import TD3
import os

# 腳本自身位置，以及往上兩層回到 rl_project 根目錄（本檔位於 experiments/evaluation/）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RL_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
_TD3_DIR = os.path.join(_RL_PROJECT_ROOT, "experiments", "td3")

# AdvancedMotorWrapper 定義在 05114（位於 experiments/td3/），需加入 sys.path 才能 import
if _TD3_DIR not in sys.path:
    sys.path.insert(0, _TD3_DIR)

# 用 importlib 引入數字開頭的模組
_mod = importlib.import_module("05114trytrain_mujoco_td3")
AdvancedMotorWrapper = _mod.AdvancedMotorWrapper

# ==========================================
# 路徑設定（全部相對於 rl_project 根目錄，與腳本實際所在子資料夾無關）
# ==========================================
LOG_DIR    = os.path.join(_RL_PROJECT_ROOT, "tensorboard_logs", "evaluation", "05121_stress_test")
OUTPUT_DIR = os.path.join(_RL_PROJECT_ROOT, "output")
MODEL_PATH = os.path.join(_RL_PROJECT_ROOT, "output", "models", "05114")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 通用環境建立函式
# ==========================================
def make_test_env(delay_steps=1, mass_scale=1.0, obs_noise_enable=False):
    """
    建立壓力測試專用環境。
    - obs_noise_enable 預設關閉，確保感測器不干擾測試變因。
    - max_episode_steps 設為 2000，讓每集有足夠的時間觀察發散行為。
    """
    base_env = gym.make("InvertedPendulum-v4", max_episode_steps=2000)
    base_env.unwrapped.model.body_mass[:] *= mass_scale
    env = AdvancedMotorWrapper(
        base_env,
        delay_steps=delay_steps,
        obs_noise_enable=obs_noise_enable,
    )
    return env


# ==========================================
# 通用 episode 評估函式
# ==========================================
def run_episode(model, env, apply_disturbance_fn=None, disturbance_start_step=500):
    """
    執行一個完整 episode，可選擇在指定步數注入擾動。

    回傳：
        survival_steps  : 存活步數
        recovery_time   : 從擾動後恢復到 |angle| < 0.01 的步數（無擾動則為 None）
        max_angle       : 整集最大角度偏移
        mean_pos_error  : 整集平均車體偏移
    """
    obs, _ = env.reset()
    survival_steps = 0
    max_angle = 0.0
    pos_errors = []
    disturbance_applied = False
    recovery_time = None
    disturbance_step = None

    for step in range(2000):
        if apply_disturbance_fn is not None and step == disturbance_start_step and not disturbance_applied:
            apply_disturbance_fn(env)
            disturbance_applied = True
            disturbance_step = step

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        survival_steps += 1

        pole_angle = obs[1]
        cart_pos   = obs[0]
        max_angle  = max(max_angle, abs(pole_angle))
        pos_errors.append(abs(cart_pos))

        if disturbance_applied and recovery_time is None and abs(pole_angle) < 0.01:
            recovery_time = step - disturbance_step

        if terminated or truncated:
            break

    mean_pos_error = float(np.mean(pos_errors))
    return survival_steps, recovery_time, max_angle, mean_pos_error


# ==========================================
# 測試一：馬達延遲掃描
# ==========================================
def test_delay(model, writer, n_episodes=50):
    print("\n" + "=" * 55)
    print("測試一：馬達延遲掃描")
    results = []

    for delay in range(1, 11):
        survival_list = []
        for _ in range(n_episodes):
            env = make_test_env(delay_steps=delay)
            steps, _, _, _ = run_episode(model, env)
            survival_list.append(steps)
            env.close()

        mean_s = np.mean(survival_list)
        std_s  = np.std(survival_list)
        results.append({"delay": delay, "mean": mean_s, "std": std_s})

        writer.add_scalar("stress/delay/mean_survival", mean_s, delay)
        writer.add_scalar("stress/delay/std_survival",  std_s,  delay)

        status = "通過" if mean_s >= 1900 else "發散"
        print(f"  Delay = {delay:>2} steps -> 平均存活 {mean_s:>7.1f} +/- {std_s:>6.1f} 步  [{status}]")

    writer.flush()
    return pd.DataFrame(results)


# ==========================================
# 測試二：初始推力干擾掃描
# ==========================================
def test_initial_disturbance(model, writer, n_episodes=50):
    print("\n" + "=" * 55)
    print("測試二：初始推力干擾掃描")
    results = []

    for disturb_mag in np.arange(0.0, 3.1, 0.3):
        survival_list = []
        for _ in range(n_episodes):
            env = make_test_env(delay_steps=1)
            obs, _ = env.reset()

            if disturb_mag > 0:
                qpos = env.env.unwrapped.data.qpos.copy()
                qvel = env.env.unwrapped.data.qvel.copy()
                qvel[1] += disturb_mag
                env.env.unwrapped.set_state(qpos, qvel)
                obs = env.env.unwrapped._get_obs()
                obs = np.concatenate((obs, env.last_real_action))

            steps, _, _, _ = run_episode(model, env)
            survival_list.append(steps)
            env.close()

        mean_s   = np.mean(survival_list)
        std_s    = np.std(survival_list)
        step_idx = round(disturb_mag * 10)
        results.append({"disturbance": round(disturb_mag, 2), "mean": mean_s, "std": std_s})

        writer.add_scalar("stress/init_disturbance/mean_survival", mean_s, step_idx)
        writer.add_scalar("stress/init_disturbance/std_survival",  std_s,  step_idx)

        status = "通過" if mean_s >= 1900 else "發散"
        print(f"  Disturbance = {disturb_mag:.1f} -> 平均存活 {mean_s:>7.1f} +/- {std_s:>6.1f} 步  [{status}]")

    writer.flush()
    return pd.DataFrame(results)


# ==========================================
# 測試三：系統質量浮動掃描
# ==========================================
def test_mass(model, writer, n_episodes=50):
    print("\n" + "=" * 55)
    print("測試三：系統質量浮動掃描")
    results = []

    for mass_pct in np.arange(0.70, 1.31, 0.05):
        survival_list = []
        for _ in range(n_episodes):
            env = make_test_env(delay_steps=1, mass_scale=mass_pct)
            steps, _, _, _ = run_episode(model, env)
            survival_list.append(steps)
            env.close()

        mean_s   = np.mean(survival_list)
        std_s    = np.std(survival_list)
        step_idx = round((mass_pct - 0.70) * 100)
        results.append({"mass_scale": round(mass_pct, 2), "mean": mean_s, "std": std_s})

        writer.add_scalar("stress/mass/mean_survival", mean_s, step_idx)
        writer.add_scalar("stress/mass/std_survival",  std_s,  step_idx)

        status = "通過" if mean_s >= 1900 else "發散"
        print(f"  Mass = {mass_pct*100:.0f}% -> 平均存活 {mean_s:>7.1f} +/- {std_s:>6.1f} 步  [{status}]")

    writer.flush()
    return pd.DataFrame(results)


# ==========================================
# 測試四 A：固定力大小，掃描擾動時長
# ==========================================
def test_disturbance_duration(model, writer, force_mag=0.5, n_episodes=50):
    print("\n" + "=" * 55)
    print("測試四 A：固定力大小，掃描擾動時長")
    results = []

    for duration in [1, 5, 10, 20, 30]:
        recovery_list = []
        survival_list = []

        for _ in range(n_episodes):
            env = make_test_env(delay_steps=1)

            def apply_disturbance(env, _dur=duration, _mag=force_mag):
                for _ in range(_dur):
                    qpos = env.env.unwrapped.data.qpos.copy()
                    qvel = env.env.unwrapped.data.qvel.copy()
                    qvel[1] += _mag
                    env.env.unwrapped.set_state(qpos, qvel)

            steps, rec_time, _, _ = run_episode(
                model, env,
                apply_disturbance_fn=apply_disturbance,
                disturbance_start_step=500
            )
            survival_list.append(steps)
            if rec_time is not None:
                recovery_list.append(rec_time)
            env.close()

        mean_rec = np.mean(recovery_list) if recovery_list else float("inf")
        mean_sur = np.mean(survival_list)
        rec_ratio = len(recovery_list) / n_episodes
        results.append({
            "duration": duration,
            "mean_recovery": mean_rec,
            "mean_survival": mean_sur,
            "recovered_ratio": rec_ratio,
        })

        writer.add_scalar("stress/disturbance_duration/mean_recovery_steps", mean_rec,   duration)
        writer.add_scalar("stress/disturbance_duration/recovery_ratio",      rec_ratio,  duration)

        print(f"  Duration = {duration:>2} steps -> 平均恢復 {mean_rec:>6.1f} 步，成功恢復率 {rec_ratio*100:.0f}%")

    writer.flush()
    return pd.DataFrame(results)


# ==========================================
# 測試四 B：固定作用時長，掃描擾動力大小
# ==========================================
def test_disturbance_force(model, writer, duration=5, n_episodes=50):
    print("\n" + "=" * 55)
    print("測試四 B：固定作用時長，掃描擾動力大小")
    results = []

    for force_mag in np.arange(0.1, 2.1, 0.2):
        recovery_list = []
        survival_list = []

        for _ in range(n_episodes):
            env = make_test_env(delay_steps=1)

            def apply_disturbance(env, _dur=duration, _mag=force_mag):
                for _ in range(_dur):
                    qpos = env.env.unwrapped.data.qpos.copy()
                    qvel = env.env.unwrapped.data.qvel.copy()
                    qvel[1] += _mag
                    env.env.unwrapped.set_state(qpos, qvel)

            steps, rec_time, _, _ = run_episode(
                model, env,
                apply_disturbance_fn=apply_disturbance,
                disturbance_start_step=500
            )
            survival_list.append(steps)
            if rec_time is not None:
                recovery_list.append(rec_time)
            env.close()

        mean_rec  = np.mean(recovery_list) if recovery_list else float("inf")
        mean_sur  = np.mean(survival_list)
        rec_ratio = len(recovery_list) / n_episodes
        step_idx  = round(force_mag * 10)
        results.append({
            "force_mag": round(force_mag, 2),
            "mean_recovery": mean_rec,
            "mean_survival": mean_sur,
            "recovered_ratio": rec_ratio,
        })

        writer.add_scalar("stress/disturbance_force/mean_recovery_steps", mean_rec,  step_idx)
        writer.add_scalar("stress/disturbance_force/recovery_ratio",      rec_ratio, step_idx)

        status = "恢復" if len(recovery_list) == n_episodes else f"{len(recovery_list)}/{n_episodes} 恢復"
        print(f"  Force = {force_mag:.1f} -> 平均恢復 {mean_rec:>6.1f} 步  [{status}]")

    writer.flush()
    return pd.DataFrame(results)


# ==========================================
# 結果彙整摘要
# ==========================================
def print_boundary_summary(df_delay, df_disturb, df_mass, df_dur, df_force):
    print("\n" + "=" * 55)
    print("模型發散邊界分析報告 (05114)")
    print("=" * 55)

    passed = df_delay[df_delay["mean"] >= 1900]
    if not passed.empty:
        print(f"  延遲容忍上限：{int(passed['delay'].max())} steps（超過此值開始發散）")
    else:
        print("  延遲：delay=1 即發散")

    passed = df_disturb[df_disturb["mean"] >= 1900]
    if not passed.empty:
        print(f"  初始推力容忍上限：{passed['disturbance'].max():.1f}（超過此值開始發散）")
    else:
        print("  初始推力：disturbance=0 即發散")

    passed = df_mass[df_mass["mean"] >= 1900]
    if not passed.empty:
        print(f"  質量容忍範圍：{passed['mass_scale'].min()*100:.0f}% ~ {passed['mass_scale'].max()*100:.0f}%")

    full_rec_dur = df_dur[df_dur["recovered_ratio"] >= 1.0]
    if not full_rec_dur.empty:
        print(f"  持續擾動時長容忍上限（force=0.5）：{int(full_rec_dur['duration'].max())} steps")

    full_rec_force = df_force[df_force["recovered_ratio"] >= 1.0]
    if not full_rec_force.empty:
        print(f"  擾動力大小容忍上限（duration=5）：{full_rec_force['force_mag'].max():.1f}")


# ==========================================
# 主程式
# ==========================================
if __name__ == "__main__":
    print("啟動 05114 模型壓力測試...")

    model  = TD3.load(MODEL_PATH)
    writer = SummaryWriter(log_dir=LOG_DIR)

    df_delay   = test_delay(model, writer)
    df_disturb = test_initial_disturbance(model, writer)
    df_mass    = test_mass(model, writer)
    df_dur     = test_disturbance_duration(model, writer)
    df_force   = test_disturbance_force(model, writer)

    print_boundary_summary(df_delay, df_disturb, df_mass, df_dur, df_force)

    # 存 CSV
    df_delay.to_csv(  os.path.join(OUTPUT_DIR, "stress_delay_results.csv"),       index=False)
    df_disturb.to_csv(os.path.join(OUTPUT_DIR, "stress_disturbance_results.csv"), index=False)
    df_mass.to_csv(   os.path.join(OUTPUT_DIR, "stress_mass_results.csv"),        index=False)
    df_dur.to_csv(    os.path.join(OUTPUT_DIR, "stress_duration_results.csv"),    index=False)
    df_force.to_csv(  os.path.join(OUTPUT_DIR, "stress_force_results.csv"),       index=False)

    # 輸出六張子圖
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("05114 Model Stress Test Results", fontsize=14, fontweight="bold")

    axes[0, 0].errorbar(df_delay["delay"], df_delay["mean"], yerr=df_delay["std"], marker="o", color="steelblue")
    axes[0, 0].axhline(1900, color="g", linestyle="--", label="Pass threshold")
    axes[0, 0].set_xlabel("Delay Steps"); axes[0, 0].set_ylabel("Mean Survival Steps")
    axes[0, 0].set_title("Test 1: Delay Scan"); axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].errorbar(df_disturb["disturbance"], df_disturb["mean"], yerr=df_disturb["std"], marker="o", color="darkorange")
    axes[0, 1].axhline(1900, color="g", linestyle="--")
    axes[0, 1].set_xlabel("Initial Disturbance"); axes[0, 1].set_ylabel("Mean Survival Steps")
    axes[0, 1].set_title("Test 2: Initial Disturbance Scan"); axes[0, 1].grid(True)

    axes[0, 2].errorbar(df_mass["mass_scale"] * 100, df_mass["mean"], yerr=df_mass["std"], marker="o", color="purple")
    axes[0, 2].axhline(1900, color="g", linestyle="--")
    axes[0, 2].set_xlabel("Mass Scale (%)"); axes[0, 2].set_ylabel("Mean Survival Steps")
    axes[0, 2].set_title("Test 3: Mass Variation Scan"); axes[0, 2].grid(True)

    axes[1, 0].plot(df_dur["duration"], df_dur["mean_recovery"], marker="s", color="crimson")
    axes[1, 0].set_xlabel("Disturbance Duration (steps)"); axes[1, 0].set_ylabel("Mean Recovery Steps")
    axes[1, 0].set_title("Test 4A: Disturbance Duration Scan"); axes[1, 0].grid(True)

    axes[1, 1].plot(df_force["force_mag"], df_force["mean_recovery"], marker="s", color="crimson")
    axes[1, 1].set_xlabel("Force Magnitude"); axes[1, 1].set_ylabel("Mean Recovery Steps")
    axes[1, 1].set_title("Test 4B: Disturbance Force Scan"); axes[1, 1].grid(True)

    axes[1, 2].plot(df_force["force_mag"], df_force["recovered_ratio"] * 100, marker="^", color="teal")
    axes[1, 2].set_xlabel("Force Magnitude"); axes[1, 2].set_ylabel("Recovery Rate (%)")
    axes[1, 2].set_title("Test 4B: Recovery Rate vs Force"); axes[1, 2].grid(True)

    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "stress_test_05114_results.png")
    plt.savefig(png_path, dpi=150)
    print(f"圖表已儲存：{png_path}")

    writer.close()
    print("\n壓力測試完成！結果已存入 TensorBoard 與 CSV。")
    print(f"  TensorBoard log : {LOG_DIR}")
    print(f"  CSV / PNG       : {OUTPUT_DIR}")

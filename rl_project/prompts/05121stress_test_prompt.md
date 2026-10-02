# Prompt for Claude Code：製作壓力測試執行檔

## 任務目標

基於已訓練完成的 `0513.zip` 模型，建立一個獨立的壓力測試執行檔 `stress_test_0513.py`。  
**不要修改任何訓練檔案，此檔案為純評估用途。**

---

## 前置說明

### 需要引入的模組

```python
from 051trytrain_mujoco_td3 import AdvancedMotorWrapper  # 引用訓練用的 Wrapper
from stable_baselines3 import TD3
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
```

### 載入模型

```python
model = TD3.load("0513")
```

### TensorBoard 輸出目錄

```python
log_dir = "rl_project/tensorboard_logs/stress_test_0513/"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)
```

### 通用環境建立函式

每個測試項目都用此函式建立乾淨的測試環境，**不帶任何訓練期的隨機擾動**：

```python
def make_test_env(delay_steps=1, mass_scale=1.0, obs_noise_enable=False):
    """
    建立壓力測試專用環境。
    - obs_noise_enable 預設關閉，確保感測器不干擾測試變因。
    - max_episode_steps 設為 2000，讓每集有足夠的時間觀察發散行為。
    """
    base_env = gym.make('InvertedPendulum-v4', max_episode_steps=2000)
    
    # 修改質量（透過 MuJoCo model 調整）
    base_env.unwrapped.model.body_mass[:] *= mass_scale
    
    env = AdvancedMotorWrapper(
        base_env,
        delay_steps=delay_steps,
        obs_noise_enable=obs_noise_enable,
    )
    return env
```

### 通用 episode 評估函式

```python
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
    angle_history = []
    disturbance_applied = False
    recovery_time = None
    disturbance_step = None

    for step in range(2000):
        # 在指定步數施加擾動
        if apply_disturbance_fn is not None and step == disturbance_start_step and not disturbance_applied:
            apply_disturbance_fn(env)
            disturbance_applied = True
            disturbance_step = step

        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        survival_steps += 1

        # 取乾淨 obs 的角度（obs 的 index 1 = pole_angle）
        pole_angle = obs[1]
        cart_pos = obs[0]
        angle_history.append(abs(pole_angle))
        max_angle = max(max_angle, abs(pole_angle))
        pos_errors.append(abs(cart_pos))

        # 計算恢復時間：擾動後第一次回到 |angle| < 0.01
        if disturbance_applied and recovery_time is None and abs(pole_angle) < 0.01:
            recovery_time = step - disturbance_step

        if terminated or truncated:
            break

    mean_pos_error = float(np.mean(pos_errors))
    return survival_steps, recovery_time, max_angle, mean_pos_error
```

---

## 測試一：馬達延遲掃描

**目的**：找出模型在馬達延遲增加到多少步時開始發散。

**變因**：`delay_steps` 從 1 步掃描到 10 步。  
**固定**：mass_scale=1.0，無初始擾動，無擾動力。  
**每個設定跑 10 集**，取平均存活步數。

```python
def test_delay(model, writer, n_episodes=10):
    print("\n" + "="*55)
    print("🔬 測試一：馬達延遲掃描")
    results = []

    for delay in range(1, 11):
        survival_list = []
        for ep in range(n_episodes):
            env = make_test_env(delay_steps=delay)
            steps, _, _, _ = run_episode(model, env)
            survival_list.append(steps)
            env.close()

        mean_s = np.mean(survival_list)
        std_s  = np.std(survival_list)
        results.append({"delay": delay, "mean": mean_s, "std": std_s})

        # 寫入 TensorBoard
        writer.add_scalar("stress/delay/mean_survival", mean_s, delay)
        writer.add_scalar("stress/delay/std_survival",  std_s,  delay)

        status = "✅ 通過" if mean_s >= 1900 else "❌ 發散"
        print(f"  Delay = {delay:>2} steps → 平均存活 {mean_s:>7.1f} ± {std_s:>6.1f} 步  {status}")

    writer.flush()
    return pd.DataFrame(results)
```

**發散門檻**：平均存活步數 < 1900 步視為發散。

---

## 測試二：初始推力干擾

**目的**：找出模型能承受的最大初始推力大小。

**變因**：初始推力從 0.0 掃描到 3.0（每次增加 0.3）。  
**固定**：delay_steps=1，mass_scale=1.0。  
**實作方式**：episode reset 後，第一步施加一次衝量到角速度上（模擬初始推桿）。  
**每個設定跑 10 集**，取平均存活步數。

```python
def test_initial_disturbance(model, writer, n_episodes=10):
    print("\n" + "="*55)
    print("🔬 測試二：初始推力干擾掃描")
    results = []

    for disturb_mag in np.arange(0.0, 3.1, 0.3):
        survival_list = []
        for ep in range(n_episodes):
            env = make_test_env(delay_steps=1)
            obs, _ = env.reset()

            # 初始推力：直接修改 MuJoCo 角速度狀態
            if disturb_mag > 0:
                qpos = env.env.unwrapped.data.qpos.copy()
                qvel = env.env.unwrapped.data.qvel.copy()
                qvel[1] += disturb_mag
                env.env.unwrapped.set_state(qpos, qvel)
                obs = env.env.unwrapped._get_obs()
                import numpy as np
                obs = np.concatenate((obs, env.last_real_action))

            steps, _, _, _ = run_episode(model, env)
            survival_list.append(steps)
            env.close()

        mean_s = np.mean(survival_list)
        std_s  = np.std(survival_list)
        results.append({"disturbance": round(disturb_mag, 2), "mean": mean_s, "std": std_s})

        # 寫入 TensorBoard（以 disturbance*10 當整數 step）
        step_idx = round(disturb_mag * 10)
        writer.add_scalar("stress/init_disturbance/mean_survival", mean_s, step_idx)
        writer.add_scalar("stress/init_disturbance/std_survival",  std_s,  step_idx)

        status = "✅ 通過" if mean_s >= 1900 else "❌ 發散"
        print(f"  Disturbance = {disturb_mag:.1f} → 平均存活 {mean_s:>7.1f} ± {std_s:>6.1f} 步  {status}")

    writer.flush()
    return pd.DataFrame(results)
```

---

## 測試三：系統質量浮動

**目的**：找出模型對系統質量變化（±30%）的容忍範圍。

**變因**：`mass_scale` 從 0.70 掃描到 1.30（每次增加 0.05）。  
**固定**：delay_steps=1，無擾動。  
**每個設定跑 10 集**，取平均存活步數。

```python
def test_mass(model, writer, n_episodes=10):
    print("\n" + "="*55)
    print("🔬 測試三：系統質量浮動掃描")
    results = []

    for mass_pct in np.arange(0.70, 1.31, 0.05):
        survival_list = []
        for ep in range(n_episodes):
            env = make_test_env(delay_steps=1, mass_scale=mass_pct)
            steps, _, _, _ = run_episode(model, env)
            survival_list.append(steps)
            env.close()

        mean_s = np.mean(survival_list)
        std_s  = np.std(survival_list)
        results.append({"mass_scale": round(mass_pct, 2), "mean": mean_s, "std": std_s})

        step_idx = round((mass_pct - 0.70) * 100)
        writer.add_scalar("stress/mass/mean_survival", mean_s, step_idx)
        writer.add_scalar("stress/mass/std_survival",  std_s,  step_idx)

        status = "✅ 通過" if mean_s >= 1900 else "❌ 發散"
        print(f"  Mass = {mass_pct*100:.0f}% → 平均存活 {mean_s:>7.1f} ± {std_s:>6.1f} 步  {status}")

    writer.flush()
    return pd.DataFrame(results)
```

---

## 測試四：抗擾動測試（Disturbance Rejection）

**目的**：在 episode 進行中途對單擺施加外力，測試模型恢復到穩定狀態所需的步數。  
**擾動施加在單擺的角速度（`qvel[1]`）**，不是馬達 action，模擬真實物理世界的外部衝擊。

**此測試分兩個子測試：**

### 4a：固定力大小，掃描作用時長

**變因**：連續施加擾動的步數（1, 5, 10, 20, 30 步）。  
**固定**：每步擾動量 = 0.5（角速度增量）、disturbance_start_step = 500。

```python
def test_disturbance_duration(model, writer, force_mag=0.5, n_episodes=10):
    print("\n" + "="*55)
    print("🔬 測試四 A：固定力大小，掃描擾動時長")
    results = []

    for duration in [1, 5, 10, 20, 30]:
        recovery_list = []
        survival_list = []

        for ep in range(n_episodes):
            env = make_test_env(delay_steps=1)

            # 定義擾動函式：連續 duration 步施加 force_mag 到角速度
            step_counter = {"count": 0}
            def apply_disturbance(env):
                for _ in range(duration):
                    qpos = env.env.unwrapped.data.qpos.copy()
                    qvel = env.env.unwrapped.data.qvel.copy()
                    qvel[1] += force_mag
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

        mean_rec = np.mean(recovery_list) if recovery_list else float('inf')
        mean_sur = np.mean(survival_list)
        results.append({
            "duration": duration,
            "mean_recovery": mean_rec,
            "mean_survival": mean_sur,
            "recovered_ratio": len(recovery_list) / n_episodes
        })

        writer.add_scalar("stress/disturbance_duration/mean_recovery_steps", mean_rec,  duration)
        writer.add_scalar("stress/disturbance_duration/recovery_ratio",      len(recovery_list)/n_episodes, duration)

        print(f"  Duration = {duration:>2} steps → 平均恢復 {mean_rec:>6.1f} 步，成功恢復率 {len(recovery_list)/n_episodes*100:.0f}%")

    writer.flush()
    return pd.DataFrame(results)
```

### 4b：固定作用時長，掃描力大小

**變因**：每步角速度擾動量從 0.1 掃描到 2.0（每次增加 0.2）。  
**固定**：持續時長 = 5 步、disturbance_start_step = 500。

```python
def test_disturbance_force(model, writer, duration=5, n_episodes=10):
    print("\n" + "="*55)
    print("🔬 測試四 B：固定作用時長，掃描擾動力大小")
    results = []

    for force_mag in np.arange(0.1, 2.1, 0.2):
        recovery_list = []
        survival_list = []

        for ep in range(n_episodes):
            env = make_test_env(delay_steps=1)

            def apply_disturbance(env):
                for _ in range(duration):
                    qpos = env.env.unwrapped.data.qpos.copy()
                    qvel = env.env.unwrapped.data.qvel.copy()
                    qvel[1] += force_mag
                    env.env.unwrapped.set_state(qpos, qvel)

            steps, rec_time, max_angle, _ = run_episode(
                model, env,
                apply_disturbance_fn=apply_disturbance,
                disturbance_start_step=500
            )
            survival_list.append(steps)
            if rec_time is not None:
                recovery_list.append(rec_time)
            env.close()

        mean_rec = np.mean(recovery_list) if recovery_list else float('inf')
        mean_sur = np.mean(survival_list)
        step_idx = round(force_mag * 10)
        results.append({
            "force_mag": round(force_mag, 2),
            "mean_recovery": mean_rec,
            "mean_survival": mean_sur,
            "recovered_ratio": len(recovery_list) / n_episodes
        })

        writer.add_scalar("stress/disturbance_force/mean_recovery_steps", mean_rec,  step_idx)
        writer.add_scalar("stress/disturbance_force/recovery_ratio",      len(recovery_list)/n_episodes, step_idx)

        status = "✅ 恢復" if len(recovery_list) == n_episodes else f"⚠️ {len(recovery_list)}/{n_episodes} 恢復"
        print(f"  Force = {force_mag:.1f} → 平均恢復 {mean_rec:>6.1f} 步  {status}")

    writer.flush()
    return pd.DataFrame(results)
```

---

## 結果彙整與輸出

訓練完四項測試後，整合成一份 CSV 報告，並輸出四張圖表：

```python
def print_boundary_summary(df_delay, df_disturb, df_mass, df_dur, df_force):
    """印出各測試的通過邊界摘要"""
    print("\n" + "="*55)
    print("📋 模型發散邊界分析報告")
    print("="*55)

    # 延遲邊界
    passed = df_delay[df_delay["mean"] >= 1900]
    if not passed.empty:
        print(f"  ✅ 延遲容忍上限：{int(passed['delay'].max())} steps（超過此值開始發散）")
    else:
        print(f"  ❌ 延遲：delay=1 即發散")

    # 初始推力邊界
    passed = df_disturb[df_disturb["mean"] >= 1900]
    if not passed.empty:
        print(f"  ✅ 初始推力容忍上限：{passed['disturbance'].max():.1f}（超過此值開始發散）")
    else:
        print(f"  ❌ 初始推力：disturbance=0 即發散")

    # 質量邊界
    passed = df_mass[df_mass["mean"] >= 1900]
    if not passed.empty:
        print(f"  ✅ 質量容忍範圍：{passed['mass_scale'].min()*100:.0f}% ~ {passed['mass_scale'].max()*100:.0f}%")

    # 抗擾邊界（恢復率 = 100%）
    full_rec_dur = df_dur[df_dur["recovered_ratio"] >= 1.0]
    if not full_rec_dur.empty:
        print(f"  ✅ 持續擾動時長容忍上限（force=0.5）：{int(full_rec_dur['duration'].max())} steps")
    full_rec_force = df_force[df_force["recovered_ratio"] >= 1.0]
    if not full_rec_force.empty:
        print(f"  ✅ 擾動力大小容忍上限（duration=5）：{full_rec_force['force_mag'].max():.1f}")
```

### 存 CSV

```python
df_delay.to_csv("stress_delay_results.csv", index=False)
df_disturb.to_csv("stress_disturbance_results.csv", index=False)
df_mass.to_csv("stress_mass_results.csv", index=False)
df_dur.to_csv("stress_duration_results.csv", index=False)
df_force.to_csv("stress_force_results.csv", index=False)
```

### 輸出圖表

四個子圖合為一張：

```python
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("0513 Model Stress Test Results", fontsize=14, fontweight='bold')

# 圖 1：延遲掃描
axes[0,0].errorbar(df_delay["delay"], df_delay["mean"], yerr=df_delay["std"], marker='o', color='steelblue')
axes[0,0].axhline(1900, color='g', linestyle='--', label='Pass threshold')
axes[0,0].set_xlabel("Delay Steps"); axes[0,0].set_ylabel("Mean Survival Steps")
axes[0,0].set_title("Test 1: Delay Scan"); axes[0,0].legend(); axes[0,0].grid(True)

# 圖 2：初始干擾掃描
axes[0,1].errorbar(df_disturb["disturbance"], df_disturb["mean"], yerr=df_disturb["std"], marker='o', color='darkorange')
axes[0,1].axhline(1900, color='g', linestyle='--')
axes[0,1].set_xlabel("Initial Disturbance"); axes[0,1].set_ylabel("Mean Survival Steps")
axes[0,1].set_title("Test 2: Initial Disturbance Scan"); axes[0,1].grid(True)

# 圖 3：質量掃描
axes[0,2].errorbar(df_mass["mass_scale"]*100, df_mass["mean"], yerr=df_mass["std"], marker='o', color='purple')
axes[0,2].axhline(1900, color='g', linestyle='--')
axes[0,2].set_xlabel("Mass Scale (%)"); axes[0,2].set_ylabel("Mean Survival Steps")
axes[0,2].set_title("Test 3: Mass Variation Scan"); axes[0,2].grid(True)

# 圖 4：抗擾時長掃描（恢復時間）
axes[1,0].plot(df_dur["duration"], df_dur["mean_recovery"], marker='s', color='crimson')
axes[1,0].set_xlabel("Disturbance Duration (steps)"); axes[1,0].set_ylabel("Mean Recovery Steps")
axes[1,0].set_title("Test 4A: Disturbance Duration Scan"); axes[1,0].grid(True)

# 圖 5：抗擾力大小掃描（恢復時間）
axes[1,1].plot(df_force["force_mag"], df_force["mean_recovery"], marker='s', color='crimson')
axes[1,1].set_xlabel("Force Magnitude"); axes[1,1].set_ylabel("Mean Recovery Steps")
axes[1,1].set_title("Test 4B: Disturbance Force Scan"); axes[1,1].grid(True)

# 圖 6：抗擾成功率
axes[1,2].plot(df_force["force_mag"], df_force["recovered_ratio"]*100, marker='^', color='teal')
axes[1,2].set_xlabel("Force Magnitude"); axes[1,2].set_ylabel("Recovery Rate (%)")
axes[1,2].set_title("Test 4B: Recovery Rate vs Force"); axes[1,2].grid(True)

plt.tight_layout()
plt.savefig("stress_test_0513_results.png", dpi=150)
print("✅ 圖表已儲存：stress_test_0513_results.png")
```

---

## 主程式執行順序

```python
if __name__ == "__main__":
    print("🚀 啟動 0513 模型壓力測試...")

    model = TD3.load("0513")
    log_dir = "rl_project/tensorboard_logs/stress_test_0513/"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 四項測試依序執行
    df_delay   = test_delay(model, writer)
    df_disturb = test_initial_disturbance(model, writer)
    df_mass    = test_mass(model, writer)
    df_dur     = test_disturbance_duration(model, writer)
    df_force   = test_disturbance_force(model, writer)

    # 邊界摘要
    print_boundary_summary(df_delay, df_disturb, df_mass, df_dur, df_force)

    # 儲存結果
    df_delay.to_csv("stress_delay_results.csv", index=False)
    df_disturb.to_csv("stress_disturbance_results.csv", index=False)
    df_mass.to_csv("stress_mass_results.csv", index=False)
    df_dur.to_csv("stress_duration_results.csv", index=False)
    df_force.to_csv("stress_force_results.csv", index=False)

    # 輸出圖表
    # （圖表程式碼如上）

    writer.close()
    print("\n✅ 壓力測試完成！結果已存入 TensorBoard 與 CSV。")
```

---

## 輸出檔案總覽

| 檔案 | 內容 |
|---|---|
| `stress_test_0513_results.png` | 六張子圖的總覽報告 |
| `stress_delay_results.csv` | 延遲掃描原始數據 |
| `stress_disturbance_results.csv` | 初始干擾掃描原始數據 |
| `stress_mass_results.csv` | 質量浮動掃描原始數據 |
| `stress_duration_results.csv` | 抗擾時長掃描原始數據 |
| `stress_force_results.csv` | 抗擾力大小掃描原始數據 |
| `rl_project/tensorboard_logs/stress_test_0513/` | TensorBoard 所有測試曲線 |

---

## TensorBoard 標籤總覽

| 標籤 | 橫軸 | 意義 |
|---|---|---|
| `stress/delay/mean_survival` | delay steps (1~10) | 各延遲設定的平均存活步數 |
| `stress/init_disturbance/mean_survival` | disturbance * 10 | 各初始干擾的平均存活步數 |
| `stress/mass/mean_survival` | (mass-0.7)*100 | 各質量設定的平均存活步數 |
| `stress/disturbance_duration/mean_recovery_steps` | duration steps | 不同時長擾動的平均恢復步數 |
| `stress/disturbance_duration/recovery_ratio` | duration steps | 不同時長擾動的成功恢復率 |
| `stress/disturbance_force/mean_recovery_steps` | force * 10 | 不同力大小的平均恢復步數 |
| `stress/disturbance_force/recovery_ratio` | force * 10 | 不同力大小的成功恢復率 |

---

## 注意事項

1. **不要修改訓練檔案**，壓力測試只載入 `0513.zip` 模型，不進行任何訓練
2. **每個測試結束後呼叫 `env.close()`**，避免 MuJoCo 環境資源洩漏
3. **擾動施加位置是 MuJoCo `qvel[1]`**（單擺角速度），不是馬達 action
4. **中文註解與 emoji 標題風格**與訓練檔案保持一致
5. `make_test_env` 的 `obs_noise_enable=False` 預設關閉，確保感測器雜訊不影響各項測試的控制變因
6. 若測試過程中發現某個參數極端值讓模型在第 1 步就倒下，仍然正常記錄（survival=1），不跳過

# Prompt for Claude Code：生成 051trytrain_mujoco_td3.py

## 任務目標

基於現有的 `041trytrain_mujoco_ddpg.py`（DDPG 架構訓練倒單擺）建立一個全新的進階訓練檔案 `051trytrain_mujoco_td3.py`，將演算法升級為 TD3 並加入多項工程強化機制。**請勿修改 041 原檔，新檔案需獨立可執行**。

## 環境資訊

- 環境：`InvertedPendulum-v4` (MuJoCo)
- 套件：`gymnasium`, `stable-baselines3`, `numpy`, `torch`
- 目錄結構需保留：`rl_project/tensorboard_logs/`

---

## 一、演算法升級：DDPG → TD3

### 1.1 引入 TD3

```python
from stable_baselines3 import TD3
```

### 1.2 模型初始化參數

```python
model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log=log_dir,
    policy_delay=2,             # Actor 每 2 步才更新一次
    target_policy_noise=0.2,    # Target Policy Smoothing 雜訊
    target_noise_clip=0.5,      # 雜訊裁切範圍
    learning_rate=1e-3,
    batch_size=256,
    buffer_size=200000,
)
```

### 1.3 探索雜訊調整

- 將 `NormalActionNoise` 的 `sigma` 從 `0.1` 調整為 `0.2`（補償 TD3 較慢的 Actor 更新）

### 1.4 訓練設定

- `total_timesteps=150000`
- `tb_log_name="TD3_Advanced_Control"`
- 模型儲存名稱：`td3_inverted_pendulum`

---

## 二、Wrapper 強化：`RealisticMotorWrapper` 升級為 `AdvancedMotorWrapper`

保留原本所有功能（延遲、低通濾波、LQR 獎勵、穩態死區獎勵），**新增以下兩大功能**：

### 2.1 OBS 觀測值加入感測器雜訊

**新增 `__init__` 參數**：

```python
obs_noise_std=0.01,          # 感測器雜訊標準差
obs_noise_enable=True,       # 是否啟用感測器雜訊
```

**實作邏輯（關鍵）**：

- LQR 獎勵計算必須使用「乾淨的 obs」（模擬真實物理狀態）
- 只有「回傳給 Agent 的 obs」才加入雜訊（模擬感測器讀數）
- `reset()` 和 `step()` 的回傳 obs 都要處理

```python
if self.obs_noise_enable:
    noisy_obs = obs + np.random.normal(0, self.obs_noise_std, size=obs.shape)
else:
    noisy_obs = obs
augmented_obs = np.concatenate((noisy_obs, self.last_real_action))
```

### 2.2 平滑度懲罰（保留原 jerk_penalty 並擴充）

**新增 `__init__` 參數**：

```python
snap_penalty=0.02,           # 二階差分懲罰（加速度變化）
smoothness_penalty=0.05,     # 移動平均偏差懲罰
```

**新增狀態記憶體**：

```python
self.last_last_raw_action = np.zeros(self.action_space.shape)
self.action_history = deque(maxlen=5)
```

**新增獎勵計算項**：

- **Snap（二階差分）**：`(action - 2*last_raw + last_last_raw)²`
- **Moving Average Deviation**：當 `action_history` 累積到 3 筆以上時，計算當前 action 與最近 5 步移動平均的偏差平方

**整合進總獎勵**：

```python
modified_reward = 1.0 - (q_angle + q_pos + q_pole_vel + q_cart_vel
                          + r_action + r_jerk + r_snap + r_smooth)
```

**注意**：別忘了在 `step()` 結尾正確更新 `self.last_last_raw_action = self.last_raw_action.copy()`，順序不能錯。

### 2.3 完整 step() 邏輯順序（請嚴格遵守）

```
1.  低通濾波 + 延遲處理 → delayed_action
2.  env.step(delayed_action) 取得乾淨 obs
3.  用「乾淨 obs」計算 LQR 獎勵（q_angle, q_pos, q_pole_vel, q_cart_vel）
4.  計算所有懲罰項（r_action, r_jerk, r_snap, r_smooth）
5.  計算穩態死區超級獎勵
6.  clip 獎勵到 [-5.0, 10.0]
7.  累積控制工程指標（見第三章）
8.  更新 last_last_raw_action ← last_raw_action
9.  更新 last_raw_action ← action
10. 更新 action_history（append）
11. 對乾淨 obs 加雜訊 → noisy_obs
12. 拼接 noisy_obs + last_real_action 回傳
```

---

## 三、控制工程性能指標監控（核心需求）

### 3.1 目的

訓練過程中，除了 `reward` 之外，使用**傳統控制工程的性能指標**來描述單擺的穩定品質。這些指標比單純的 reward 更直觀反映系統的控制性能，方便從機電／控制工程角度判斷模型訓練狀況。

**這不是另外做一個控制器做比較，而是把控制工程指標套用到 TD3 自身的訓練過程上，作為訓練品質的描述工具。**

### 3.2 需要追蹤的指標

在 `AdvancedMotorWrapper` 內加入 episode-level 的指標累積器，**每個 episode 結束時**計算以下指標並寫入 TensorBoard：

| 指標代號 | 全名 | 計算方式 | TensorBoard 標籤 |
|---|---|---|---|
| **ISE** | Integral of Squared Error | `Σ pole_angle²` | `control/ISE` |
| **IAE** | Integral of Absolute Error | `Σ \|pole_angle\|` | `control/IAE` |
| **ITAE** | Integral of Time × Abs Error | `Σ t·\|pole_angle\|`（t 為步數） | `control/ITAE` |
| **Overshoot** | 最大角度偏移 | `max(\|pole_angle\|)` 整集中 | `control/Overshoot_max_angle` |
| **Settling Time** | 穩定時間 | 第一次進入 `\|angle\| < 0.02` 的步數；若整集未達成則記錄 episode 總長度 | `control/Settling_Time_steps` |
| **Steady-State Error** | 穩態誤差 | 該 episode 後 20% 步數的 `mean(\|pole_angle\|)` | `control/Steady_State_Error` |
| **RMS Control Effort** | 控制力均方根 | `sqrt(mean(action²))` 整集中 | `control/RMS_Control_Effort` |
| **Position ISE** | 車體位置平方積分 | `Σ cart_pos²` | `control/Position_ISE` |

### 3.3 實作方式

**Step 1：在 `__init__` 中加入累積器宣告**

```python
self._reset_metrics()
```

**Step 2：新增 `_reset_metrics()` 方法**

```python
def _reset_metrics(self):
    self.metric_angle_history = []      # 存每步 pole_angle（乾淨值）
    self.metric_pos_history = []        # 存每步 cart_pos（乾淨值）
    self.metric_action_history = []     # 存每步 delayed_action
    self.metric_step_count = 0          # episode 步數計數
    self.metric_settling_time = None    # 穩定時間（None 表示尚未達成）
```

**Step 3：在 `reset()` 中呼叫**

```python
def reset(self, **kwargs):
    # ... 原有邏輯 ...
    self._reset_metrics()
    # ...
```

**Step 4：在 `step()` 的第 7 步累積數據（使用乾淨 obs）**

```python
self.metric_step_count += 1
self.metric_angle_history.append(pole_angle)
self.metric_pos_history.append(cart_pos)
self.metric_action_history.append(float(delayed_action[0]))

# 偵測穩定時間：第一次進入 ±0.02 弧度區間
if self.metric_settling_time is None and abs(pole_angle) < 0.02:
    self.metric_settling_time = self.metric_step_count
```

**Step 5：episode 結束時計算指標並透過 `info` 字典回傳**

當 `terminated or truncated` 為 True 時：

```python
if terminated or truncated:
    info["control_metrics"] = self._compute_episode_metrics()
```

```python
def _compute_episode_metrics(self):
    angles  = np.array(self.metric_angle_history)
    positions = np.array(self.metric_pos_history)
    actions = np.array(self.metric_action_history)
    n = len(angles)

    metrics = {
        "ISE":                float(np.sum(angles ** 2)),
        "IAE":                float(np.sum(np.abs(angles))),
        "ITAE":               float(np.sum(np.arange(1, n + 1) * np.abs(angles))),
        "Overshoot_max_angle":float(np.max(np.abs(angles))),
        "Settling_Time_steps":float(self.metric_settling_time if self.metric_settling_time else n),
        "Steady_State_Error": float(np.mean(np.abs(angles[int(n * 0.8):]))) if n >= 5 else 0.0,
        "RMS_Control_Effort": float(np.sqrt(np.mean(actions ** 2))),
        "Position_ISE":       float(np.sum(positions ** 2)),
    }
    return metrics
```

### 3.4 透過 Callback 寫入 TensorBoard

stable-baselines3 內部訓練迴圈不會自動把 wrapper 的 `info` 寫進 TensorBoard，需自訂 callback。

**新增 `ControlMetricsCallback` 類別**：

```python
from stable_baselines3.common.callbacks import BaseCallback

class ControlMetricsCallback(BaseCallback):
    """
    從 env info 提取控制工程指標，每個 episode 結束時寫入 TensorBoard。
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "control_metrics" in info:
                metrics = info["control_metrics"]
                self.episode_count += 1
                for key, value in metrics.items():
                    self.logger.record(f"control/{key}", value)
                self.logger.dump(self.num_timesteps)  # 強制 flush，確保每集寫入
        return True
```

**在 `model.learn()` 時傳入**：

```python
callback = ControlMetricsCallback()
model.learn(
    total_timesteps=150000,
    tb_log_name="TD3_Advanced_Control",
    callback=callback
)
```

### 3.5 訓練後 TensorBoard 解讀方式

| 指標曲線 | 健康訓練應呈現的趨勢 |
|---|---|
| `control/ISE`、`control/IAE` | 隨訓練下降 → 控制誤差變小 |
| `control/ITAE` | 隨訓練下降 → 收斂速度加快（時間加權，越早收斂分數越低） |
| `control/Settling_Time_steps` | 隨訓練下降 → 系統收斂變快 |
| `control/Overshoot_max_angle` | 隨訓練下降 → 超調減少 |
| `control/Steady_State_Error` | 趨近 0 → 穩態精度提升 |
| `control/RMS_Control_Effort` | 隨訓練下降 → 馬達負擔變輕、動作更精省 |
| `control/Position_ISE` | 隨訓練下降 → 車體越來越能維持在中心 |

---

## 四、檔案結構模板

```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import os

# ==========================================
# 1. AdvancedMotorWrapper
#    （含感測器雜訊、平滑度懲罰、控制工程指標累積）
# ==========================================
class AdvancedMotorWrapper(gym.Wrapper):
    # ... 依上述需求 2、3 實作

# ==========================================
# 2. ControlMetricsCallback
#    （控制工程指標寫入 TensorBoard）
# ==========================================
class ControlMetricsCallback(BaseCallback):
    # ... 依上述需求 3.4 實作

# ==========================================
# 3. 主訓練程式
# ==========================================
if __name__ == "__main__":
    # 3.1 建立環境
    # 3.2 建立 TD3 模型（含 ControlMetricsCallback）
    # 3.3 開始訓練
    # 3.4 儲存模型為 "td3_inverted_pendulum"
```

---

## 五、特別注意事項

1. **變數命名一致**：所有與原 041 檔案對應的變數名稱請保留（如 `last_raw_action`、`last_real_action`、`action_queue`），方便對照閱讀。

2. **中文註解保留**：041 檔案使用中文註解風格（🚀、🌟 等 emoji 標題），請延續此風格。

3. **指標計算用乾淨 obs**：所有控制工程指標必須使用「未加雜訊的 obs」計算，因為這代表真實物理狀態。雜訊只在回傳給 Agent 時才加。

4. **不要刪除任何原檔功能**：低通濾波 `alpha`、延遲 `delay_steps`、所有 LQR 權重、穩態死區獎勵都必須完整保留。

5. **參數預設值請依照原檔保持相容**：
   ```
   delay_steps=1, alpha=0.6, force_penalty=0.005,
   jerk_penalty=0.02, position_penalty=0.5, velocity_penalty=0.1
   ```

6. **後續會有壓力測試腳本**：`AdvancedMotorWrapper` 的所有參數（`delay_steps`、`obs_noise_std`、`obs_noise_enable` 等）必須能透過 `__init__` 從外部注入，方便壓力測試腳本動態修改。`obs_noise_enable=False` 開關務必保留。

---

## 六、預期產出

一份完整、可直接執行的 `051trytrain_mujoco_td3.py`，執行後應該：

1. 開始 TD3 訓練（150,000 steps），訓練過程同時寫入 reward 與控制工程指標
2. 訓練完成後儲存模型為 `td3_inverted_pendulum`
3. TensorBoard 中可同時觀察：
   - 預設指標：`rollout/ep_rew_mean`、`rollout/ep_len_mean`
   - 自訂控制指標：`control/ISE`、`control/IAE`、`control/ITAE`、`control/Settling_Time_steps`、`control/Overshoot_max_angle`、`control/Steady_State_Error`、`control/RMS_Control_Effort`、`control/Position_ISE`

請在檔案開頭加上簡短的功能說明註解區塊，列出此版本相對於 041 的所有改進點。

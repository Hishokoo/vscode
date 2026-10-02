# 任務：以 05114 為基底建立 07111trytrain_mujoco_td3.py（Sim-to-Real 領域隨機化重校）

## 要建立的檔案

`07111trytrain_mujoco_td3.py`

**不要修改 `05114trytrain_mujoco_td3.py`**，它是壓力測試的比較基準，必須保持可回溯。

---

## 背景說明

05114 通過了壓力測試（delay=2 存活 1874 步、質量 70–130% 全過），但它的領域隨機化設定與真實硬體有嚴重錯配。實測分析結論：

**過於苛刻的部分**（浪費訓練容量，而且會讓策略過度預判）

| 參數 | 現值 | 實體推估 | 倍率 |
|------|------|----------|------|
| `delay_steps=2` | 80 ms | 編碼器+MCU 約 2–10 ms | 過度 8–40× |
| `alpha=0.6` | τ=43.7 ms（= 1.09 個控制步） | 電流控制驅動器 < 5 ms | 過度 ~10× |
| `obs_noise` 角度 | ±0.573° | 2000 CPR 編碼器 0.045° | 過度 ~12× |
| `obs_noise` 位置 | ±1 cm | 線性編碼器 < 0.1 mm | 過度 ~100× |

`delay_steps` 與 `alpha` 其實在模擬同一件事（作動滯後），兩者疊加後總滯後是 124 ms ≈ 3.1 個控制步，而桿件不穩定時間常數 τ = 202 ms — 已佔 0.61 τ。

**完全沒有模擬的部分**（上機真正的失敗來源）

- `dof_frictionloss = [0, 0]` — 滑軌與轉軸的乾摩擦是零，只有黏滯阻尼
- 感測器零點偏移（只有零均值高斯雜訊，沒有常數 bias）
- 作動器增益誤差（`gear=100` 被當成精確值）
- 延遲抖動（每集重抽、集內固定，不是逐步抖動）

---

## 一、清理：刪除兩個死參數

`position_penalty` 與 `velocity_penalty` 在 `__init__` 賦值後**從未被讀取**，真正生效的是 `step()` 裡硬寫的 `q_pos = 5.0` 與 `q_cart_vel = 0.1`。直接刪掉這兩個參數與對應的 `self.xxx = xxx` 兩行，避免日後有人調了它們卻以為有效果。

---

## 二、`__init__` 參數表

```python
def __init__(self, env,
             delay_steps=1, alpha=0.6,
             force_penalty=0.005, jerk_penalty=0.02,
             obs_noise_std=(0.001, 0.001, 0.02, 0.02), obs_noise_enable=True,
             snap_penalty=0.02, smoothness_penalty=0.05,
             delay_randomize=False, delay_random_range=(0, 1),
             disturbance_enable=False, disturbance_probability=1.0,
             disturbance_start_range=(50, 300),
             disturbance_magnitude_range=(0.3, 0.8),
             disturbance_duration_range=(3, 8),
             # ===== 🌟 以下為 07111 新增 =====
             alpha_randomize=False, alpha_range=(0.85, 0.98),
             friction_randomize=False,
             friction_slide_range=(0.0, 4.0),
             friction_hinge_range=(0.0, 0.01),
             angle_bias_enable=False, angle_bias_range=(-0.01, 0.01),
             gain_error_enable=False, gain_error_range=(0.85, 1.15),
             jitter_enable=False, jitter_prob=0.10):
```

`__init__` 內需新增的狀態變數：

```python
# 🌟 必須在任何修改之前記下原始齒輪比，否則每集會累乘而爆掉
self._base_gear = float(env.unwrapped.model.actuator_gear[0, 0])

self._friction_slide = 0.0
self._friction_hinge = 0.0
self._angle_bias     = 0.0
self._gain_error     = 1.0
self._last_applied_action = np.zeros(self.action_space.shape)

# obs_noise_std 允許純量或長度 4 的向量，統一轉成 np.array
self.obs_noise_std = np.asarray(obs_noise_std, dtype=np.float64)
```

---

## 三、新增 `_observe()` 方法（消除重複程式碼）

現在 `reset()` 與 `step()` 各有一份「加雜訊 → 串接出力」的重複邏輯，加上 bias 之後會變成三處重複。抽成一個方法：

```python
def _observe(self, clean_obs):
    """
    把物理引擎的乾淨 obs 轉成 agent 實際看到的觀測。
    順序：零點偏移 → 感測器雜訊 → 串接馬達實際出力
    """
    measured = np.asarray(clean_obs, dtype=np.float64).copy()
    measured[1] += self._angle_bias          # 🌟 bias 只污染角度通道
    if self.obs_noise_enable:
        measured = measured + np.random.normal(0.0, self.obs_noise_std, size=measured.shape)
    return np.concatenate((measured, self.last_real_action))
```

`reset()` 與 `step()` 的結尾都改成 `augmented_obs = self._observe(obs)`。

> **關鍵原則：bias 與雜訊只影響 agent 的觀測，絕對不能影響獎勵與指標。**
> `step()` 裡計算 LQR 獎勵、死區獎勵、以及所有 `control/` 指標時，一律使用物理引擎回傳的**乾淨 obs 與真實角度**。理由：bias 模擬的是「感測器讀錯」，訓練訊號必須獎勵它把**真實的桿子**立直，否則等於在獎勵它配合自己的量測誤差。

---

## 四、`reset()` 新增四段隨機化

放在既有的「初始擾動 ±0.2 rad」之後、`action_queue` 重建之前：

```python
unwrapped = self.env.unwrapped

# 🌟 1. 乾摩擦隨機化（絕對賦值，不會累乘）
if self.friction_randomize:
    self._friction_slide = np.random.uniform(*self.friction_slide_range)
    self._friction_hinge = np.random.uniform(*self.friction_hinge_range)
    unwrapped.model.dof_frictionloss[0] = self._friction_slide   # 滑軌，單位 N
    unwrapped.model.dof_frictionloss[1] = self._friction_hinge   # 轉軸，單位 N·m

# 🌟 2. 作動器增益誤差（務必以 _base_gear 為基準，不要讀當前值再乘）
if self.gain_error_enable:
    self._gain_error = np.random.uniform(*self.gain_error_range)
    unwrapped.model.actuator_gear[0, 0] = self._base_gear * self._gain_error

# 🌟 3. 感測器零點偏移
self._angle_bias = (np.random.uniform(*self.angle_bias_range)
                    if self.angle_bias_enable else 0.0)

# 🌟 4. 馬達一階響應隨機化
if self.alpha_randomize:
    self.alpha = float(np.random.uniform(*self.alpha_range))
```

同時在既有的狀態重置區加上：

```python
self._last_applied_action = np.zeros(self.action_space.shape)
```

`delay_randomize` 的抽樣邏輯沿用 05114 的 `np.random.randint(lo, hi + 1)` 即可 —
`delay_random_range=(0, 1)` 在現有 deque 邏輯下是正確的：`maxlen=0+1=1` 時 append 後立刻 popleft，等於零延遲。

---

## 五、`step()` 新增指令抖動

把原本的：

```python
smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
self.last_real_action = smoothed_action
self.action_queue.append(smoothed_action)
delayed_action = self.action_queue.popleft()
```

改為：

```python
smoothed_action = self.alpha * action + (1 - self.alpha) * self.last_real_action
self.last_real_action = smoothed_action
self.action_queue.append(smoothed_action)
new_cmd = self.action_queue.popleft()

# 🌟 指令抖動：模擬封包遲到 / 掉幀，驅動器沿用上一個值
if self.jitter_enable and np.random.random() < self.jitter_prob:
    applied_action = self._last_applied_action.copy()
else:
    applied_action = new_cmd
self._last_applied_action = applied_action.copy()
```

接著把 `step()` 中**所有** `delayed_action` 出現處改成 `applied_action`，共三處：

1. `obs, _, terminated, truncated, info = self.env.step(applied_action)`
2. `r_action = self.force_penalty * np.sum(np.square(applied_action))`
3. `self.metric_action_history.append(float(applied_action[0]))`

> 這個模型比「每步重抽 delay_steps」更貼近實體：真實系統在指令遲到時的行為就是**保持前一個值**，而不是改變管線長度。也避免中途改 deque 長度造成指令重複或遺漏。

---

## 六、指標：記錄每集實際抽到的隨機參數

在 `_compute_episode_metrics()` 回傳的 dict 尾端加上五項，方便日後用 TensorBoard 交叉比對「哪一種組合會失敗」：

```python
"dr_delay_steps":    float(self.delay_steps),
"dr_alpha":          float(self.alpha),
"dr_friction_slide": float(self._friction_slide),
"dr_angle_bias":     float(self._angle_bias),
"dr_gain_error":     float(self._gain_error),
```

`ControlMetricsCallback` 不需改動，這些會自動寫成 `control/dr_xxx`。

---

## 七、獎勵函數：一個字都不要改

`q_angle=10.0`、`q_pos=5.0`、`q_pole_vel=0.5`、`q_cart_vel=0.1`、
`force_penalty=0.005`、`jerk_penalty=0.02`、`snap_penalty=0.02`、`smoothness_penalty=0.05`、
死區獎勵 `+2.0 / +5.0`、clip 範圍 `[-5.0, 10.0]` —— **全部維持與 05114 完全相同**。

理由：這次只動領域隨機化。獎勵函數一起改的話，07111 跟 05114 的比較就同時有兩個變因，壓力測試結果無法歸因。

---

## 八、主訓練程式

```python
if __name__ == "__main__":
    base_env = gym.make('InvertedPendulum-v4', max_episode_steps=2000)
    env = AdvancedMotorWrapper(
        base_env,
        # --- 作動路徑：大幅放寬（原本過於苛刻）---
        delay_randomize=True,  delay_random_range=(0, 1),
        alpha_randomize=True,  alpha_range=(0.85, 0.98),

        # --- 感測：重新標定 ---
        obs_noise_std=(0.001, 0.001, 0.02, 0.02),
        angle_bias_enable=True, angle_bias_range=(-0.01, 0.01),

        # --- 新增的實體效應（原本完全沒有）---
        friction_randomize=True,
        friction_slide_range=(0.0, 4.0),
        friction_hinge_range=(0.0, 0.01),
        gain_error_enable=True, gain_error_range=(0.85, 1.15),
        jitter_enable=True,     jitter_prob=0.10,

        # --- 外力擾動：沿用 05114，不動 ---
        disturbance_enable=True,
        disturbance_probability=0.3,
        disturbance_start_range=(50, 300),
        disturbance_magnitude_range=(0.1, 0.3),
        disturbance_duration_range=(2, 5),
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))

    log_dir = "rl_project/tensorboard_logs/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("output/models", exist_ok=True)

    # 觀測空間仍為 5 維，與 05114 權重相容
    model = TD3.load("rl_project/05114", env=env, verbose=1, tensorboard_log=log_dir)
    model.action_noise = action_noise

    callback = ControlMetricsCallback()
    model.learn(
        total_timesteps=200000,
        tb_log_name="07111_Sim2Real_DR",
        callback=callback,
        reset_num_timesteps=True,
    )

    model.save("output/models/07111")
```

TD3 超參數（`policy_delay=2`、`target_policy_noise=0.2`、`target_noise_clip=0.5`、
`learning_rate=1e-3`、`batch_size=256`、`buffer_size=200000`）全部隨 `TD3.load` 繼承，不另外指定。

---

## 九、執行時的中止條件（重要）

06111 的教訓：從 05113 載入後換到更嚴苛的環境，造成災難性遺忘，`ep_len_mean` 從 1034 掉到 13.2，後面花了 9 輪 1.35M steps 在填洞。

這次的作動環境**比 05114 更寬鬆**（延遲與濾波都降低），理論上不該崩，但新增的摩擦與 bias 是新變因。因此：

> **訓練開始後盯著前 20K steps 的 `rollout/ep_len_mean`。
> 若掉到 100 以下且沒有回升趨勢，立刻中止**，改成不載入 05114、從隨機初始化訓練 300K steps。

不要像 0611 那樣讓它自己跑滿再檢討。

---

## 十、注意事項

- 檔名：`07111trytrain_mujoco_td3.py`，放在 `rl_project/` 根目錄
- TensorBoard log 名稱：`07111_Sim2Real_DR`，輸出至 `rl_project/tensorboard_logs/`
- 模型儲存：`output/models/07111`（**遵守 CLAUDE.md 的輸出路徑規定**，若資料夾不存在要先建立）
- 保留 `_on_training_end` 回收最後一集數據的機制，與 05114 相同
- `AdvancedMotorWrapper` 與 `ControlMetricsCallback` 目前在七支檔案裡各複製一份。這次先照做，但如果要順手處理技術債，正確做法是抽到 `tools/motor_wrapper.py` 讓各腳本 import（05121 已示範可用 `importlib` 引入）

---

## 配套工作（不在本次任務範圍，另開 prompt）

07111 訓完之後**無法用現有的 05121 驗證是否變好** —— 現行壓力測試完全沒有摩擦與 bias 這兩個變因，而那正是這次改動的重點。需要另外擴充 `05121stress_test.py`：

1. 新增「靜摩擦掃描」：`dof_frictionloss[0]` 從 0 掃到 8 N
2. 新增「零點偏移掃描」：angle bias 從 0 掃到 ±0.02 rad
3. 修掉測試二的 bug：目前先 `env.reset()` 注入推力後又呼叫 `run_episode()`，而後者開頭再 reset 一次把擾動整個沖掉，導致該測試實際上什麼都沒測到（11 個推力值結果毫無趨勢即為證據）
4. 測試四 A／四 B 的 `apply_disturbance` 迴圈內沒有推進物理，所以「持續 N 步」等價於瞬間注入 N×mag 的衝量 —— 若要測真正的持續性外力，必須改成跨步施加

---

## 🔒 設計變更：改為從零訓練，完全獨立（2026-09-24）

原設計是 `TD3.load("output/models/05114")` 續訓。為了讓 07111 **只要有原始碼就能完整重現**、
不依賴任何既有 `.zip`，已改為從隨機初始化訓練。

| 項目 | 續訓版（原） | 獨立版（現） | 為什麼變 |
|------|--------------|--------------|----------|
| 起點 | `TD3.load("output/models/05114")` | `TD3("MlpPolicy", env, ...)` | 移除對 .zip 的依賴 |
| `learning_starts` | 10000 | 100（SB3 預設） | 續訓保護，用來避免空 buffer 對已收斂策略做更新；從零沒有策略要保護 |
| `learning_rate` | 3e-4 | 1e-3 | 同上，降學習率是為了保護舊策略 |
| `sigma` | 0.1 | 0.2 | 同上，隨機初始化需要充分探索 |
| `total_timesteps` | 200K | 300K | 從零開始，且環境比 05111 多了摩擦/bias/增益誤差/抖動四層隨機化 |
| `reset_num_timesteps` | `True` | 移除 | 從零訓練不需要 |

超參數（`lr=1e-3`、`sigma=0.2`、`batch=256`、`buffer=200K`、`policy_delay=2`、
`target_policy_noise=0.2`、`target_noise_clip=0.5`）**刻意與 05111 完全一致** ——
05111 是 TD3 家族唯一從零訓練的版本，對齊它可以確保「從零訓練」本身不引入新變因。

> ⚠️ **解讀壓力測試時必須注意**：與 05114 的比較從「只差領域隨機化」
> 變成「**領域隨機化 + 訓練起點都不同**」。05114 是 05113 續訓 150K 的產物，
> 07111 是從零 300K。若兩者表現有差距，不能直接歸因於領域隨機化設定。
> 若日後需要嚴格的單一變因對照，正確做法是**另外跑一支從零 + 05114 舊 DR 設定的對照組**。

**已驗證**：在一個 `.zip` 檔案數為 0 的全新暫存目錄執行本腳本，可正常啟動並開始訓練
（`buffer size at start = 0`、`lr=0.001`、`sigma=[0.2]`），確認無任何模型檔依賴。

附帶影響：`# TODO(07112)` 裡「維持 5 維觀測以相容 05114 權重」這個約束**已不再成立**，
07112 若要改觀測空間（例如加 `VecFrameStack`）不需再顧慮權重載入。

---

## 已套用的 A 組修正（依 prompts/07111_code_review.md，2026-09-24）

上述規格完成後，另做了一輪程式碼檢查，套用了四項不破壞「與 05114 單一變因」的修正：

| 項目 | 改動 | 理由 |
|------|------|------|
| 初始擾動 | `±0.2` → `±0.15` | 終止條件是 `\|θ\|>0.2`，環境自帶 ±0.01 reset 雜訊疊加後，**實測 1.21% 的 episode 開局即死**。這些樣本與策略無關卻會進 buffer，且 delay=1 的 episode 首步必送 0 指令、在邊界帶死亡率更高，會讓 `dr_delay_steps` 帶上假訊號 |
| Settling Time | 新增 `Settling_Time_sustained` | 舊的 `Settling_Time_steps` 是「首次進入 ±0.01」，控制工程定義應為「此後持續停留在誤差帶內的最早時刻」。震盪系統穿越零點時會瞬間滿足，數字過度樂觀（實測案例：舊定義 31 步 vs 新定義 80 步） |
| ~~續訓保護~~ **（已作廢）** | ~~`learning_starts=10000`、`lr` 1e-3→3e-4、`sigma` 0.2→0.1~~ | 當時的理由成立（`TD3.load()` 實測不載入 replay buffer、05114 存檔 `learning_starts=100`，空 buffer + 已收斂策略 = 06111 災難性遺忘配方），但**改為從零訓練後已全部移除** —— 沒有收斂策略需要保護。見上一節 |
| 防呆診斷 | `n==0` 保護、啟動列印生效超參數、`initial_angle_abs` 指標 | 不改行為，讓問題早點暴露 |

> ⚠️ **Settling Time 的比較注意事項**：`Settling_Time_steps`（舊定義）**刻意保留不動**，
> 所以它與 05114 仍可直接比較。新增的 `Settling_Time_sustained` 是 07111 才有的，
> 05114 沒有對應欄位，**不要拿這兩個不同定義的數字互相比較**。

B 組（angle_bias 量級、作動鏈順序、第 5 維語意、`r_action` raw/applied 統一、位置終止條件）
會改變獎勵函數或觀測語意，已在程式碼對應位置加上 `# TODO(07112):` 註解說明問題與預定改法，
本輪**未實際修改**，以免破壞對照實驗。

---

## 可選的第二階段（07112，先不要做）

分析指出「模擬裡沒有摩擦」與「獎勵懲罰高頻小幅動作」兩個效應會疊加：`jerk_penalty` + `snap_penalty` + `smoothness_penalty` 壓抑的正是能突破靜摩擦的小幅修正指令。

但這屬於獎勵函數改動，如果和 07111 一起做就無法歸因。等 07111 的壓力測試結果出來之後，再視情況單獨降低這三項權重做消融實驗。

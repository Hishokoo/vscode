# 07111trytrain_mujoco_td3.py 程式碼檢查報告

> 這份文件給執行修改的 Claude：每一項都附了「問題 / 證據 / 建議改法 / 驗證方式」。
> 標示 **[已實測]** 的數字是在 `gymnasium` + `mujoco 3.14` 的 `InvertedPendulum-v4` 上實際跑出來的，
> 不是估算。標示 **[待確認]** 的請執行者自行驗證後再動手。
>
> **重要前提**：這個腳本是 `TD3.load("output/models/05114")` 續訓，設計目的是與 05114
> 做「只有領域隨機化不同」的對照實驗。因此**任何會改變獎勵函數或觀測語意的修改，都會破壞這個單一變因**。
> 下面的建議已按「可否在本輪安全套用」分組，請嚴格遵守分組，不要把 B 組的東西併進這一輪。

---

## 0. 已驗證的環境事實（修改前請以這些為準）

以下皆由 `InvertedPendulum-v4` 原始碼與實機模擬確認：

| 項目 | 數值 | 來源 |
|---|---|---|
| 終止條件 | `abs(obs[1]) > 0.2` rad（**只有角度，沒有位置終止**） | `inverted_pendulum_v4.py` L40 |
| 時間步長 `dt` | 0.04 s（timestep 0.02 × frame_skip 2） | 實測 `env.unwrapped.dt` |
| 環境自帶 reset 雜訊 | qpos、qvel 皆 `uniform(-0.01, 0.01)` | `reset_model()` |
| 動作範圍 | `ctrlrange = [-3, 3]`，`gear = 100` → **最大出力 300 N** | XML actuator |
| 質量 | cart 10.47 kg、pole 5.02 kg，合計 15.49 kg | 實測 `model.body_mass` |
| 滑軌 | `slider` joint `range="-1 1"`, `limited="true"` → **±1 m 硬限位** | XML / 實測 |
| 關節阻尼 | slider 與 hinge 皆 `damping=1` | XML default |
| 桿長 | 0.6 m | XML geom |

SB3 `TD3` 建構子預設值（v2.9.0 確認）：
`learning_rate=1e-3, buffer_size=1e6, learning_starts=100, batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5`

⚠️ 但本腳本是 `load()`，**實際生效的是 05114 存檔裡的超參數，不是上表預設值**。
修改第 7 項前，請先執行：

```python
m = TD3.load("output/models/05114")
for k in ["learning_rate", "learning_starts", "batch_size", "tau", "gamma",
          "train_freq", "gradient_steps", "policy_delay", "buffer_size"]:
    print(k, getattr(m, k, None))
print("net_arch:", m.policy_kwargs)
```

---

## 修改分組

**A 組（本輪可套用，不破壞與 05114 的單一變因）**
- 第 1 項：初始擾動範圍
- 第 6 項：Settling Time 定義
- 第 7 項：續訓超參數
- 第 9 項：診斷輸出（新增，不改行為）

**B 組（概念上該改，但會改變獎勵或觀測語意，本輪請勿套用，只留 TODO 註解）**
- 第 2 項：angle_bias 與死區量級
- 第 3 項：作動鏈順序
- 第 4 項：第 5 維語意 / frame stacking
- 第 5 項：r_action 的 raw/applied 統一
- 第 8 項：位置終止條件

---

# A 組

## 1. 初始擾動範圍撞到終止邊界 ⭐ 最高優先

**位置**：`reset()`，約 L176

```python
qpos[1] += np.random.uniform(-0.2, 0.2)
qvel[1] += np.random.uniform(-0.1, 0.1)
```

**問題**：終止條件是 `|角度| > 0.2`。環境自帶 ±0.01 的 reset 雜訊，再加上你的 ±0.2，
合成後的初始角度可以超過 0.2。

**[已實測]** 20000 次取樣的合成初始角度分布：

| 事件 | 機率 |
|---|---|
| `\|θ₀\| > 0.2`（一開局就已終止） | **1.40 %** |
| `\|θ₀\| > 0.18`（距離邊界 < 0.02） | **10.31 %** |
| `\|θ₀\| > 0.15` | 25.61 % |

**[已實測]** 零出力自由落體，從 θ₀ 到超過 0.2 需要幾步：

| θ₀ | 步數 |
|---|---|
| 0.199 | 1 |
| 0.19 | 2 |
| 0.18 | 3 |
| 0.15 | 4 |

**需要修正我先前的說法**：作動器非常強（300 N / 15.5 kg），**這些局面並非不可救**。
實測從 θ₀ = 0.195、ω = 0.11 施加滿出力，三步內角度就被拉回 −0.15，沒有終止。
所以「近邊界 = 必死」是錯的。

真正的問題有兩個，而且都會污染你的判讀：

1. **1.4 % 的 episode 長度為 1，且與策略完全無關。** 這些樣本進了 buffer，教 critic
   「這一區價值極低」，但那是初始化造成的，不是策略造成的。
2. **[已實測] 與 `delay_steps` 產生耦合。** 延遲 = 1 的 episode，第一步送出的指令必定是 0
   （佇列裡預填的零）。從 θ₀ = 0.195、ω = 0.11 出發，那一步無控自由落體後角度到 0.2038 →
   **當場終止**；θ₀ = 0.185 則存活（0.1936）。
   也就是說，延遲 = 1 的 episode 在邊界帶的死亡率明顯高於延遲 = 0。
   你的 `dr_delay_steps` 指標會因此帶上一個與策略無關的偏差，
   讓「延遲隨機化造成策略變差」這個結論被假訊號放大。

而你在註解裡指定要盯的 `rollout/ep_len_mean`，正好是被這兩件事污染的那個指標。

**建議改法**：

```python
# 修改前
qpos[1] += np.random.uniform(-0.2, 0.2)
# 修改後（保留 0.04 rad 安全邊界，約等於 10 步的自由落體餘裕）
qpos[1] += np.random.uniform(-0.15, 0.15)
```

`qvel[1]` 的 ±0.1 不必動，該量級沒有問題。

**驗證方式**：改完後跑 5000 次 reset，確認 `max(|qpos[1]|) < 0.17`，且沒有任何 episode
在第一步就 terminated：

```python
import numpy as np, gymnasium as gym
env = AdvancedMotorWrapper(gym.make('InvertedPendulum-v4', max_episode_steps=2000), ...)
mx = 0; dead = 0
for _ in range(5000):
    obs, _ = env.reset()
    mx = max(mx, abs(env.unwrapped.data.qpos[1]))
    _, _, term, _, _ = env.step(np.zeros(1))
    dead += int(term)
print(mx, dead)   # 期望：mx < 0.17, dead == 0
```

---

## 6. Settling Time 的定義與名稱不符

**位置**：`step()` L328

```python
if self.metric_settling_time is None and abs(pole_angle) < 0.01:
    self.metric_settling_time = self.metric_step_count
```

**問題**：這是「**首次進入**誤差帶」。控制工程的 settling time 定義是
「**此後持續停留在**誤差帶內的最早時刻」。震盪系統穿越零點時會瞬間滿足 `< 0.01`，
於是得到一個過度樂觀、而且與 ITAE 走向可能矛盾的數字。

另外閾值 0.01 與 `angle_bias` 上限相等（見第 2 項），這個指標實際上大半由當集抽到的 bias 決定。

**建議改法**：刪掉 step() 裡的那段，改在 `_compute_episode_metrics()` 裡回溯計算。
閾值同時放寬到 0.02，與獎勵死區一致：

```python
# 在 _compute_episode_metrics() 裡，n 算出來之後加入：
SETTLE_BAND = 0.02
outside = np.where(np.abs(angles) > SETTLE_BAND)[0]
settling = float(outside[-1] + 1) if len(outside) > 0 else 0.0
if len(outside) > 0 and outside[-1] == n - 1:
    settling = float(n)      # 直到最後都沒穩定，記為整集長度
```

然後把字典裡那一行換掉：

```python
"Settling_Time_steps": settling,
```

`step()` 裡的 `metric_settling_time` 相關兩行、以及 `_reset_metrics()` 裡的
`self.metric_settling_time = None` 可以一併移除。

**注意**：這會讓 07111 的 `Settling_Time_steps` 與 05114 的**不可直接比較**（定義變了）。
請在 commit message 與 `prompts/07111_sim2real_dr.md` 裡明確記下這件事。
如果你需要與 05114 對照，建議**兩個都記**：保留舊的叫 `Settling_Time_first_entry`，
新的叫 `Settling_Time_sustained`。這是比較安全的做法。

**驗證方式**：手工餵一段人造角度序列（例如前 50 步在帶外、之後都在帶內），
確認回傳 51。再餵一段「進帶又出帶」的序列，確認回傳的是最後一次出帶之後的索引。

---

## 7. 續訓的超參數沒有跟著調整 ⭐ 這是 06111 災難的直接成因

**位置**：`__main__` L431–449

**問題**：`TD3.load()` **不會載入 replay buffer**（需要另外呼叫 `load_replay_buffer()`）。
所以訓練開始時的狀態是：

- policy：已在較寬鬆環境收斂的 05114 權重
- buffer：**空的**
- `learning_starts`：若沿用預設 100，代表**第 101 步就開始用極少量新樣本做梯度更新**
- `action_noise`：`sigma=0.2`，對一個已收斂的策略而言偏大

一個收斂策略 + 一批高度相關的少量樣本 + 正常學習率 = 災難性遺忘的標準配方。
這正是 06111 `ep_len_mean` 從 1034 掉到 13.2 的機制。你現有的對策（盯著看、不對就中止）
是事後偵測；下面是事前預防。

**建議改法**：

```python
model = TD3.load(
    "output/models/05114",
    env=env,
    verbose=1,
    tensorboard_log=log_dir,
    # === 續訓保護 ===
    learning_starts=10000,   # 先用舊策略在新環境跑滿 10K 步填 buffer，再開始更新
    learning_rate=3e-4,      # 從 1e-3 降一檔
)
model.action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)   # 0.2 → 0.1
)
```

**[已驗證] SB3 的 `load()` 確實支援用 kwargs 覆寫**：原始碼中
`model.__dict__.update(data)` 之後緊接 `model.__dict__.update(kwargs)`，然後才
`model._setup_model()`。因為 `_setup_model()` 會呼叫 `_setup_lr_schedule()` 重建
learning rate schedule，所以 `learning_rate` 用 kwargs 傳入是有效的。
（若遇到版本相容問題，改用 `custom_objects={"learning_rate": 3e-4}` 也可以。）

`sigma` 由 0.2 降到 0.1 是折衷：新環境確實需要探索（摩擦、增益誤差是全新的），
但對收斂策略施加過大的探索雜訊會直接破壞它。若 10K 步後 `ep_len_mean` 仍在高檔，
可以考慮回調到 0.15。

**驗證方式**：啟動後確認 log 中 `learning_starts` 生效——前 10K 步不應出現
`train/actor_loss`、`train/critic_loss`。並在 15K、30K 步檢查 `rollout/ep_len_mean`
是否維持在 05114 的水準附近（不應出現斷崖）。

---

## 9.（新增）加上防呆與診斷輸出

這一項不改變任何訓練行為，純粹是讓問題早點暴露。

**9a. `_compute_episode_metrics()` 的除零防護**

`n = len(angles)`，多處使用 `/ n`。目前的呼叫路徑保證 `n >= 1`，但 `_on_training_end`
裡的回收路徑依賴 `metric_step_count > 0` 這個外部檢查。建議在函式開頭直接加：

```python
if n == 0:
    return {}
```

順便把 `float(self.metric_settling_time if self.metric_settling_time else n)`
這種依賴 truthiness 的寫法改掉（若值為 0 會誤判），用 `is not None` 判斷。

**9b. 在啟動時印出實際生效的超參數**

```python
print(f"lr={model.learning_rate}  learning_starts={model.learning_starts}  "
      f"batch={model.batch_size}  train_freq={model.train_freq}  "
      f"sigma={model.action_noise._sigma if model.action_noise else None}")
print(f"buffer size after load = {model.replay_buffer.size()}")   # 應為 0，確認 buffer 未載入
```

**9c. 記錄每集是否為「開局即死」**

在 `_compute_episode_metrics()` 的字典裡加一行，方便事後把污染樣本濾掉：

```python
"initial_angle_abs": float(abs(self.metric_angle_history[0])),
```

---

# B 組（本輪請勿套用，只加 TODO 註解）

以下每一項都請在程式碼對應位置加上 `# TODO(07112):` 註解，說明問題與預定改法，
但**不要實際修改**，以免破壞與 05114 的對照實驗。

## 2. angle_bias 量級與獎勵死區互相侵蝕

**位置**：L402 `angle_bias_range=(-0.01, 0.01)`；L314 死區判定

**需要修正我先前的說法**：我原本寫「+2 的獎勵有很大一部分由抽籤決定」，**這是錯的**。
實際上死區半寬 0.02 > bias 上限 0.01，所以 agent 只要把量測讀數壓到 0，
真實角度必定落在 `|−b| ≤ 0.01 < 0.02` 內，**+2 的獎勵一定拿得到**。

真正的代價是**餘裕不對稱，而且 agent 不知道偏向哪一邊**：

| 當集 bias | 量測負方向餘裕 | 量測正方向餘裕 |
|---|---|---|
| b = 0 | 0.02 | 0.02 |
| b = +0.01 | **0.01** | 0.03 |
| b = −0.01 | 0.03 | **0.01** |

agent 唯一的理性應對是假設最壞情況、對齊到兩邊都安全的 ±0.01，
也就是學出比必要更保守的策略。這本身不算壞（實機上保守通常是對的），
但要知道這是無意間加上的要求。

**預定改法（07112）**：`angle_bias_range` 縮到 ±0.005，或死區放寬到 0.04，兩者擇一。

## 3. 作動鏈的物理順序反了

**位置**：`step()` L250–260

目前順序：`低通濾波 → 延遲佇列 → jitter → MuJoCo`
物理正確順序：`延遲（通訊/運算）→ jitter（掉幀 hold）→ 低通濾波（馬達慣性）→ 出力`

低通濾波模擬的是馬達電感與轉子慣性，位於整條鏈的**最下游**，應排在延遲與 jitter 之後。

具體不自洽：jitter 觸發時驅動器 hold 住舊值，但 `last_real_action`（濾波器狀態）
已經按新指令更新過了，hold 的效果完全沒進到馬達動態。真實硬體上掉幀會讓馬達
持續朝舊目標爬升，這個效應目前模擬不到。

**預定改法（07112）**：三段順序對調，先進佇列 → popleft → jitter → 最後對取出的指令做濾波。
注意這會連帶改變第 5 維的定義，需與第 4 項一起處理。

## 4. 第 5 維的語意含糊且跨 episode 漂移

**位置**：L108 註解、L167 `np.concatenate((measured, self.last_real_action))`

註解寫「馬達真實出力」，實際是**濾波後、進延遲佇列前**的指令，與真正作用在滑車上的力
相差延遲、jitter、gain_error、摩擦四層。至少應更正註解。

更實質的問題：這一維的物理意義隨當集抽到的 `delay_steps` 而變——
延遲 = 0 時它只是當步推力的 2–15 % 成分（權重 1−α）；延遲 = 1 時它**就是**當步推力的全部。
而 policy 是無記憶前饋網路，看不到 `delay_steps`，也無法從單步 obs 判斷自己在哪個模式。
同一個數字在不同 episode 代表不同物理量，對函數近似是不小的負擔。

**預定改法（07112）**，二選一：
- 加 `VecFrameStack(n_stack=4)`，讓網路能從「下指令到狀態反應」的時間差推斷延遲；
- 或把 `delay_steps` 固定為 1（真實系統的延遲通常穩定，不會每次開機都變），
  把隨機化預算挪到更值得的地方。

附帶說明：`angle_bias` 在原理上是可以從長期動態推出來的（倒立擺只有真實角度 0
才是可長期維持的平衡點，所以撐得夠久時量測讀數會收斂到 θ̃ ≈ b），但無記憶前饋網路
取不到這個資訊。所以目前的設定實際上是在訓練「對未知零點偏移保持穩健」，
而不是「估測並補償零點偏移」。兩個目標都合理，只是要知道自己在做哪一個。

## 5. 獎勵裡 raw / applied 混用

**位置**：`step()` L295–306

```python
r_action = self.force_penalty * np.sum(np.square(applied_action))   # applied
r_jerk   = self.jerk_penalty  * np.sum(np.square(action - self.last_raw_action))   # raw
r_snap   = ...                                                       # raw
r_smooth = ...                                                       # raw
```

`r_action` 用 applied，代表 jitter 觸發的那 10 % 步數裡，agent 因為一個它沒下、
也無法預測的指令被罰。這筆懲罰是純噪音，只會推高 TD loss 的下限。

另外 `RMS_Control_Effort` 記錄的是 applied，與上述三個懲罰項不同源，交叉比對時要注意。

**預定改法（07112）**：`r_action` 也改用 `action`，四項懲罰統一針對網路自身輸出。
實體出力的代價已經由 gain_error 與摩擦間接反映在狀態裡了。

## 8. 缺少位置終止條件（sim-to-real 缺口）

**需要修正我先前的說法**：我原本說「滑車可以無限往一邊跑」，**這是錯的**。
**[已實測]** XML 裡 `slider` 是 `limited="true" range="-1 1"`，MuJoCo 以約束方式強制執行；
持續施加 100 N 向外推，cart_pos 停在 1.0008 m，確實有硬限位。

實際的問題是**沒有位置終止**：agent 撞到 ±1 m 的限位後不會被終止，可以靠著牆繼續撐桿子，
而且那個狀態在模擬裡是「安全」的。真實硬體上那就是撞限位開關或撞機構。
目前唯一約束位置的只有 `q_pos = 5.0 * cart_pos²` 這個軟懲罰，agent 完全可以為了撐住桿子
而接受位置代價一路滑出去。

**預定改法（07112）**，在 `step()` 步驟 8 之後加入：

```python
RAIL_HALF = 0.9          # 依實際滑軌長度設定，需小於 MuJoCo 的 1.0
if abs(cart_pos) > RAIL_HALF:
    terminated = True
```

這比任何雜訊隨機化都更接近「真實世界的約束」。但因為它會顯著改變 episode 長度分布，
必須與 05114 重新做對照，不能在本輪加入。

---

# 附錄：一個值得記錄的觀察（不需修改）

**乾摩擦在小出力區造成作動死區。**

`friction_slide_range=(0.0, 4.0)` 相對於最大出力 300 N 看起來微不足道（1.3 %），
但在平衡點附近，控制量本來就很小，這時 4 N 是決定性的。

**[已實測]** 固定指令 5 N（action = 0.05）連續 25 步，滑車位移：

| `dof_frictionloss[0]` | 25 步後 cart_pos | cart_vel |
|---|---|---|
| 0.0 N | 0.2527 m | 0.376 |
| 1.0 N | 0.2150 m | 0.413 |
| 4.0 N | **0.0564 m** | 0.173 |

也就是說，當 `|指令| < 4 N`（即 `|action| < 0.04`）時，滑車幾乎推不動。
這與獎勵裡的 `q_pos = 5.0 * cart_pos²` 直接衝突——要把位置歸零需要精細的小出力，
但小出力正好落在摩擦死區裡。

這不是 bug，而是**你想模擬的真實效應本來就是這樣**（真實滑軌確實有靜摩擦）。
但它解釋了為什麼加入摩擦隨機化後，`Position_ISE` 很可能會明顯變差，
而那不代表策略退化。請在分析時把 `dr_friction_slide` 當作分組變數，
而不是把所有 episode 混在一起看平均。

---

# 執行檢查清單

- [ ] 先跑第 0 節的超參數列印，確認 05114 存檔裡實際的數值
- [ ] 套用第 1 項（初始擾動 0.2 → 0.15），跑驗證腳本確認 `dead == 0`
- [ ] 套用第 6 項（Settling Time 回溯計算），建議新舊兩個指標並存
- [ ] 套用第 7 項（`learning_starts=10000`、`lr=3e-4`、`sigma=0.1`）
- [ ] 套用第 9 項（防呆與診斷輸出），確認 `replay_buffer.size() == 0`
- [ ] 在第 2、3、4、5、8 項對應位置加上 `# TODO(07112):` 註解，**不做實際修改**
- [ ] 在檔頭註解區更新「相較於 05114 的改進」清單，加入本次 A 組的四項
- [ ] 更新 `prompts/07111_sim2real_dr.md`，特別註明 Settling Time 定義變更一事
- [ ] 啟動訓練，在 15K 與 30K 步檢查 `rollout/ep_len_mean` 無斷崖

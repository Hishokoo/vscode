# 07113 修改規格：純 RL 改善滑車位置穩定性

> 給 Claude Code 的實作說明。請依本文件從 `experiments/machine/07112trytrain_mujoco_td3.py` 複製出 `experiments/machine/07113trytrain_mujoco_td3.py` 再修改，**不要動 07112 原檔**。

## 0. 背景與目標

07112 的結果（300K 步、seed 0、每組評估 5 集）：

| 指標（名目模型、尾段） | RL | LQI | 倍數 |
|---|---|---|---|
| 評估 return | 2,690 | 5,349 | LQI 高 2 倍 |
| x 標準差 | 7.61 cm | 1.24 cm | 6.1× |
| θ 標準差 | 14.3 mrad | 8.5 mrad | 1.7× |
| RMS duty | 0.34 | 0.13 | 2.6× |
| duty 飽和比例 | 3.2% | 0% | — |

關鍵判讀：**在同一個獎勵函數下 LQI 的 return 是 RL 的兩倍**，代表獎勵本身已經偏好「滑車不動」，問題在 TD3 沒有把這個獎勵最佳化到位。因此這一版**不改獎勵**，改的是評估、探索、時間尺度、動作平滑性、critic 輸入。

**目標**：部署時仍是純神經網路策略（不疊加 LQI、不做 residual RL）。07112 腳本開頭寫的「07113 改 residual RL」計畫作廢，07113 改為本文件內容。

**成功判準**：名目模型下 `osc_x_std_tail` 的 RL / LQI 倍數 ≤ 2，且 `angle_limit_hit` 不高於 07112。

---

## 總體實作原則

1. **每一項改動都做成命令列參數**，可以單獨開關，方便 ablation。預設值要能重現 07112 的行為（見第 7 節的 `--preset 07112`）。
2. 不要用 `VecNormalize`，觀測縮放維持 `ObservationBuilder` 內的固定常數（上機共用）。
3. TensorBoard run 名稱與輸出檔名要包含設定標籤與 seed，例如 `07113_E2_25hz_s1`，避免互相覆蓋。
4. **輸出路徑**：07112 是從 repo 根目錄 `vscode/` 執行，輸出跑到 `vscode/output/` 而不是 `rl_project/` 底下。請改成以腳本檔案位置推出專案根目錄（`Path(__file__).resolve()` 往上找），讓 `output/`、`tensorboard_logs/` 不論從哪裡執行都落在同一處。先確認實際的專案根目錄結構再決定往上幾層。

---

## 1. 評估修正（必做，其餘所有改動都依賴它）

### 問題
- 每次評估只有 5 集，成功率刻度是 20%，「4/5 vs 5/5」只差一集，雜訊太大。
- `07112_best.zip` 是用這 5 集的 DR return 挑出來的，挑存檔點本身帶有運氣成分。
- 只跑一個訓練 seed，無法判斷改動是否真的有效。
- 評估成本：每 25K 步評估 2 環境 × 集數 × 最多 1000 步，集數拉高後評估步數會超過訓練步數。

### 改法
1. **訓練中評估**：`--eval-freq` 預設改 50,000；`--eval-episodes` 預設改 10。種子維持 `10_000` 起。
2. **訓練結束的最終評估**（新增）：
   - 對 `final` 模型與 `best` 模型各跑 `--final-eval-episodes`（預設 50）集，名目與全開 DR 各一組。
   - **種子必須和訓練中評估錯開**，使用 `20_000` 起。原因：best 是用 10000 那組挑出來的，再用同一組打分數會偏高。
   - LQI 也在同一組 20000 種子上跑，當作對照。
   - 結果寫成 `output/results/{run_name}_final_eval.json`，內容包含：設定參數（全部 argparse 參數）、seed、`summarize()` 的輸出（RL final、RL best、LQI，各自 nominal / dr），以及每集的原始 metrics（方便事後重算）。
3. **多 seed**：`--seed` 已存在；確認所有存檔名都帶 seed。
4. **新增彙整腳本** `experiments/machine/07113_aggregate.py`：
   - 讀取 `output/results/*_final_eval.json`，依設定標籤分組。
   - 每組輸出各指標的「跨 seed 平均 ± 標準差」，以及對 LQI 的倍數。
   - 至少列出：`full_length_frac`、`return_mean`、`osc_x_std_tail`、`osc_theta_std_tail`、`osc_theta_zc_hz_tail`、`max_cart_pos_abs`、`RMS_Control_Effort`、`Duty_Saturation_frac`、`rail_limit_hit`、`angle_limit_hit`。
   - 輸出 markdown 表格到 stdout，並另存 `output/results/07113_summary.md`。

---

## 2. 探索雜訊調整

### 問題
探索雜訊 σ=0.2 duty，比 LQI 平衡時的 RMS duty（0.13）還大；`target_policy_noise=0.2` 讓 critic 在 ±0.2 duty 範圍內把 Q 平均掉，抹平精細的位置修正。每步獨立的白雜訊在 50 Hz 下對 x 這種慢模態幾乎沒有探索效果（Raffin et al., CoRL 2021, gSDE）。

### 改法（新增參數）
| 參數 | 07112 值 | 07113 預設 |
|---|---|---|
| `--explore-sigma` | 0.2 | 0.08 |
| `--explore-sigma-final` | 0.05 | 0.02 |
| `--noise-type` | normal | `normal`（可選 `ou`） |
| `--target-noise` | 0.2 | 0.05 |
| `--target-noise-clip` | 0.5 | 0.1 |

- `NoiseDecayCallback` 改用 `--explore-sigma` → `--explore-sigma-final`，時間排程（前 60% 固定、60%→90% 線性下降）不變。
- `--noise-type ou`：使用 SB3 的 `OrnsteinUhlenbeckActionNoise(mean=0, sigma=..., theta=0.15, dt=1e-2)`。`NoiseDecayCallback` 修改 `_sigma` 的方式對 OU 也適用，請確認屬性名稱一致。

---

## 3. 時間尺度：控制頻率可切換（50 Hz / 25 Hz）

### 問題
每個動作只維持 20 ms，對慢模態 x 的影響太小，「好動作」與「普通動作」的 Q 值差距被 critic 的估計誤差淹沒（Tallec et al., ICML 2019：DDPG 類方法在時間步變小時會崩潰）。θ 是快模態所以學得起來，x 學不起來，與觀察吻合。

### 改法
新增 `--ctrl-hz {50,25}`，預設 50。

**⚠️ 實作陷阱**：07112 把 `CTRL_DT`、`N_SUBSTEPS`、`MAX_EPISODE_STEPS` 寫成模組層級常數，而且有地方把它們當**函式預設參數**，例如：

```python
class ObservationBuilder:
    def __init__(self, dt=CTRL_DT, ...):      # 預設值在 import 時就綁定了
class IP570Env(gym.Env):
    def __init__(self, ..., max_episode_steps=MAX_EPISODE_STEPS):
```

如果只是在 `main()` 裡改全域變數，這些預設值**不會跟著變**。請改成一個時間設定物件（例如 `dataclass Timing`），由 `main()` 依 `--ctrl-hz` 建立後**明確傳入**環境、builder、LQI 設計函式、callbacks、`summarize()`。不要依賴 import 時綁定的預設值。

需要跟著頻率變動的項目：

| 項目 | 50 Hz | 25 Hz | 說明 |
|---|---|---|---|
| `N_SUBSTEPS` | 10 | 20 | `PHYS_DT` 維持 2 ms |
| `CTRL_DT` | 0.02 | 0.04 | |
| `MAX_EPISODE_STEPS` | 1000 | 500 | 一集固定 20 秒 |
| `gamma` | 0.995 | 0.99 | 公式 `gamma = 0.99 ** (CTRL_DT / 0.04)`，維持約 4 秒視野 |
| `DISTURBANCE["start_range"]` | (100, 750) | (50, 375) | 單位是步，改成以秒定義：2–15 s |
| `DISTURBANCE["duration_range"]` | (2, 5) | (1, 3) | 改成以秒定義：40–100 ms，換算後取整、至少 1 步 |
| `summarize()` 的跑滿判定 | `>= 1000` | `>= 500` | 改用傳入的 max steps |
| `_compute_episode_metrics` 的 `n >= 100` | | | 改成 `n >= 2 秒對應的步數` |

自動跟著變、不需特別處理（但請確認有收到正確的 dt）：
- `ObservationBuilder` 的速度低通 α 與積分 dt
- `_linearize_nominal()` 的子步數（LQI 會在 25 Hz 下重新設計，`latency_substeps=3` 維持）
- 通訊延遲 2–10 ms 仍在週期內的前幾個子步
- 外力大小以 `impulse / (dur * CTRL_DT)` 計算，衝量不變

`07113_config.json` 要記錄實際使用的 `ctrl_dt`、`n_substeps`、`gamma`。**上機時 MCU/PC 的控制週期必須與訓練一致。**

### 比較方式
25 Hz 的 300K 步等於 50 Hz 的兩倍模擬時間。請在 TensorBoard 額外記錄 `time/sim_seconds = num_timesteps * CTRL_DT`，事後可同時比較「相同步數」與「相同模擬時間」。比較時看 **RL 對同頻率 LQI 的倍數**，不是絕對值。

---

## 4. 動作平滑：CAPS 正則化

### 問題
RL 用了約 2.6 倍的出力卻控制得更差，代表大量出力在「推過頭再拉回」，配合 80 ms 的機械時間常數形成 x 擺盪。07112 的 jerk/snap 懲罰在獎勵裡，必須先經過 critic 學會才能影響 actor，路徑太間接。

### 方法（Mysore et al., ICRA 2021, CAPS）
在 actor loss 直接加入兩項：

```
actor_loss = −Q(s, π(s)) / mean|Q|  +  λ_T · ‖π(s_t) − π(s_{t+1})‖  +  λ_S · ‖π(s) − π(s + ε)‖,   ε ~ N(0, σ_caps²)
```

- Q 項除以 `|Q|` 的 batch 平均（`.detach()`），讓 λ 不受 Q 量級影響（做法借自 TD3+BC）。
- 時間項用 `(1 − dones)` 遮罩，集結束的轉移不要求平滑。
- 獎勵中的 jerk/snap 懲罰**保留不動**，維持獎勵與 07112 相同。

### 新增參數
| 參數 | 預設 | 說明 |
|---|---|---|
| `--caps-lambda-t` | 0.0 | 0 = 關閉 |
| `--caps-lambda-s` | 0.0 | 0 = 關閉 |
| `--caps-sigma` | 0.02 | 加在**縮放後觀測**上的雜訊，約等於 θ 4 mrad、x 3 mm |

兩個 λ 皆為 0 時，請直接用原版 `TD3`，確保關閉時行為與原版完全一致。

### 實作
繼承 `TD3`，覆寫 `train()`。**請先打開目前安裝的 `stable_baselines3/td3/td3.py`，以該版本的 `train()` 為底**，只在 actor 更新處加入 CAPS 項，其餘（critic 更新、polyak、batch norm stats 更新、logger）保持與原版一致。參考骨架（以 SB3 2.x 為準）：

```python
class TD3CAPS(TD3):
    def __init__(self, *args, lambda_t=0.0, lambda_s=0.0, caps_sigma=0.02, **kwargs):
        self.lambda_t, self.lambda_s, self.caps_sigma = lambda_t, lambda_s, caps_sigma
        super().__init__(*args, **kwargs)

    # train() 中 actor 更新段改為：
    #   obs = rd.observations
    #   a = self.actor(obs)
    #   q = self.critic.q1_forward(obs, a)
    #   q_term = -q.mean() / q.abs().mean().detach()
    #   a_next = self.actor(rd.next_observations)
    #   l_t = ((a - a_next).norm(dim=1) * (1 - rd.dones.squeeze(1))).mean()
    #   a_near = self.actor(perturb(obs, self.caps_sigma))
    #   l_s = (a - a_near).norm(dim=1).mean()
    #   actor_loss = q_term + self.lambda_t * l_t + self.lambda_s * l_s
```

- `perturb()` 必須能處理第 5 節的 dict 觀測：**只對 `"obs"` 加雜訊，`"priv"` 不動**。
- 額外 log：`train/caps_temporal`、`train/caps_spatial`、`train/q_abs_mean`。
- 注意：Q 項正規化只在 CAPS 開啟時使用。

### 掃描範圍
λ_T = λ_S ∈ {0.1, 0.3, 1.0}。λ 過大會讓策略遲鈍、倒擺增加，必須同時看 `angle_limit_hit`（特別是 DR 環境的外力擾動情境）。

---

## 5. Asymmetric critic（特權資訊 critic）

### 問題
critic 只看到 6 維觀測：速度經差分 + 低通而落後真實值，且看不到每集抽到的物理參數。同一個觀測在不同參數的機器上對應不同的未來，critic 只能估一個吵雜的平均值，∂Q/∂a 不準，x 的精細控制學不起來。

### 方法（Pinto et al., RSS 2018）
actor 維持 6 維觀測（上機用），critic 額外看模擬器的真實狀態與參數。critic 不會上機，因此不受實機限制。

### 新增參數
`--asym-critic`（flag，預設關閉）。

### 環境端
`IP570Env(..., privileged_obs=False)`。開啟時觀測空間改為：

```python
gym.spaces.Dict({
    "obs":  Box(shape=(6,)),         # ObservationBuilder 輸出，與 07112 完全相同
    "priv": Box(shape=(N_PRIV,)),
})
```

`priv` 內容（全部正規化到約 [−1, 1]）：

| 項目 | 來源 | 正規化 |
|---|---|---|
| 真實 x, θ, ẋ, θ̇ | `data.qpos`, `data.qvel` | 除以 `OBS_SCALE` 對應值 |
| K_F, B, armature, m_pole, lc | `self.q`（`derive_plant` 的輸出） | `log(值 / 名目值)` |
| Fc_cart, b_cart, hinge_fric, hinge_damp | `self.p` | `(值 − 中點) / 半寬`，用 `PLANT_DR` 範圍 |
| angle_bias, angle_scale, adc_noise, enc_cal | `self.sens` | 用 `SENSOR_DR` 範圍做同樣正規化 |
| 本步實際延遲（子步數） | step 中抽到的 `lat` | `(lat − 中點) / 半寬` |
| 外力 | 本步施加的 `F` | 除以名目最大外力（由 `DISTURBANCE` 最大 Δθ̇ 與最短持續時間算出） |

名目模型（DR 關閉）時這些欄位自然為 0 或固定值，不需特殊處理。

### 網路端
SB3 TD3 預設 actor 與 critic 不共用特徵抽取器，可分別指定：

```python
class ObsOnly(BaseFeaturesExtractor):          # actor：只取 obs
    def __init__(self, space):
        super().__init__(space, features_dim=space["obs"].shape[0])
    def forward(self, o):
        return o["obs"]

class ObsPriv(BaseFeaturesExtractor):          # critic：obs + priv
    def __init__(self, space):
        super().__init__(space, features_dim=space["obs"].shape[0] + space["priv"].shape[0])
    def forward(self, o):
        return th.cat([o["obs"], o["priv"]], dim=1)

class AsymTD3Policy(TD3Policy):
    def make_actor(self, features_extractor=None):
        kw = self._update_features_extractor(self.actor_kwargs, ObsOnly(self.observation_space))
        return Actor(**kw).to(self.device)

model = TD3(AsymTD3Policy, env, policy_kwargs=dict(features_extractor_class=ObsPriv), ...)
```

- 請確認 `make_critic` 使用 `ObsPriv`、`make_actor`（含 actor_target）使用 `ObsOnly`；`share_features_extractor` 必須為 False。
- SB3 遇到 dict 觀測會自動用 `DictReplayBuffer`，確認即可。
- 需跟著改的地方：`PeriodicEvalCallback` 與最終評估的環境也要開 `privileged_obs`；LQI 控制器讀 `env.builder`，不受影響。
- **與 CAPS 併用**：`TD3CAPS` 要能接受 `AsymTD3Policy`，空間平滑雜訊只加在 `"obs"`。

### 部署
上機時只需要 actor，`ObsOnly` 是恆等映射，所以可直接：

```python
with th.no_grad():
    duty = model.actor.mu(th.as_tensor(obs6[None], dtype=th.float32)).numpy()[0]
```

請額外匯出 `output/models/{run_name}_actor.pt`（只含 `actor.mu` 的 state_dict 與網路結構描述），並在 `07113_config.json` 註明 `asym_critic: true` 與部署方式。

### 已知限制
critic 知道 actor 推不出來的資訊時，理論上可能高估策略表現（部分可觀測下純狀態不對稱 critic 的偏差）。本案 actor 有 `u_applied` 與 `∫x`，部分效應可間接觀測，影響應較小。判斷依據：`eval_dr` 與 `eval_nominal` 的差距是否縮小。

---

## 6. 不要改的東西

- 獎勵函數（`REWARD`、`_well`、clip）全部維持 07112。
- 物理參數、DR 範圍、感測與通訊模型維持 07112。
- 終止條件（θ 0.2 rad、x 0.15 m）維持。
- 觀測縮放 `OBS_SCALE` 與 6 維觀測的定義維持（actor 端）。

---

## 7. 命令列介面總覽

```
--preset {none,07112}         07112 = 重現舊設定（σ 0.2→0.05, target 0.2/0.5, 50 Hz, 無 CAPS, 無 asym）
--tag STR                     設定標籤，用於 run 名稱與輸出檔名
--seed INT
--timesteps INT               預設 300000
--ctrl-hz {50,25}             預設 50
--eval-freq INT               預設 50000
--eval-episodes INT           預設 10
--final-eval-episodes INT     預設 50
--explore-sigma FLOAT         預設 0.08
--explore-sigma-final FLOAT   預設 0.02
--noise-type {normal,ou}      預設 normal
--target-noise FLOAT          預設 0.05
--target-noise-clip FLOAT     預設 0.1
--caps-lambda-t FLOAT         預設 0
--caps-lambda-s FLOAT         預設 0
--caps-sigma FLOAT            預設 0.02
--asym-critic                 flag
--lqi-only                    沿用 07112
```

`--preset 07112` 套用後，若同時有明確指定的參數，以明確指定者為準。

run 名稱格式：`07113_{tag}_{ctrl_hz}hz_s{seed}`。

---

## 8. 實驗計畫

每個設定跑 seed 0、1、2。可以多開行程平行跑（各 seed 獨立程序），注意 CPU 核心數。

| 代號 | 設定 | 要回答的問題 |
|---|---|---|
| E0 | `--preset 07112` | 07112 在新評估下的真實水準（基準線） |
| E1 | 新雜訊參數（預設值） | 探索雜訊是不是主因 |
| E2 | E1 + `--ctrl-hz 25` | 時間尺度是不是主因 |
| E3a/b/c | E1 或 E2 中較好者 + CAPS λ = 0.1 / 0.3 / 1.0 | 平滑化能否壓低出力與 x 擺盪 |
| E4 | E3 中最佳 + `--asym-critic` | critic 資訊不足是不是主因 |

決策規則：
- 依序進行，每一階段用彙整腳本比較「跨 seed 平均 ± 標準差」。
- 進步需大於跨 seed 標準差才算數。
- 任一階段名目 `osc_x_std_tail` 倍數 ≤ 2 且 `angle_limit_hit` 未上升，即達成目標，後續階段改為「是否還能再進步」的探索。
- 新增 `experiments/machine/07113_run_all.sh`（或 `.py`），依上表產生並執行所有指令；支援只跑指定代號。

---

## 9. 實作完成後的驗證清單（請 Claude Code 逐項確認）

1. **冒煙測試**：每個代號以 `--timesteps 20000 --eval-freq 10000 --eval-episodes 2 --final-eval-episodes 2` 跑通，無例外。
2. **形狀檢查**：關閉 asym 時觀測 shape 為 (6,)；開啟時為 dict，`obs` (6,)、`priv` (N_PRIV,)。
3. **頻率檢查**：25 Hz 下印出 `CTRL_DT=0.04`、`MAX_EPISODE_STEPS=500`、`gamma≈0.99`，且 `ObservationBuilder.dt`、LQI 線性化子步數都是 25 Hz 的值（不是 import 時的預設值）。
4. **LQI 檢查**：`--lqi-only --ctrl-hz 50` 的 LQI 增益應與 07112 報告一致：`[-6.109, -11.854, -4.823, -1.978, -0.059, -2.307]`。
5. **CAPS 關閉一致性**：λ 皆為 0 時確認使用原版 `TD3`。
6. **asym 部署檢查**：用匯出的 `actor.pt` 對 6 維觀測推論，結果與 `model.predict(dict_obs, deterministic=True)` 相同。
7. **種子隔離**：最終評估使用 20000 起的種子，與訓練中評估（10000 起）不重疊。
8. **輸出位置**：從 repo 根目錄與從腳本目錄各執行一次，確認輸出都落在同一個專案輸出資料夾。
9. 更新腳本開頭的說明註解：說明 07113 與 07112 的差異、各參數的意義，並移除「07113 改 residual RL」的舊計畫描述。

---

## 參考文獻

- Mysore et al., *Regularizing Action Policies for Smooth Control with Reinforcement Learning* (CAPS), ICRA 2021. arXiv:2012.06644
- Raffin, Kober, Stulp, *Smooth Exploration for Robotic Reinforcement Learning* (gSDE), CoRL 2021. arXiv:2005.05719
- Tallec, Blier, Ollivier, *Making Deep Q-learning Methods Robust to Time Discretization*, ICML 2019. arXiv:1901.09732
- Pinto et al., *Asymmetric Actor Critic for Image-Based Robot Learning*, RSS 2018. arXiv:1710.06542
- Chen et al., *Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies*, 2024. arXiv:2410.11825（CAPS 的替代方案）

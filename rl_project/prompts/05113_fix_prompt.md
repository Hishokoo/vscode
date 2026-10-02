# Prompt for Claude Code：修正 051trytrain_mujoco_td3.py 的三個問題

## 任務目標

修改現有的 `051trytrain_mujoco_td3.py`，針對 TensorBoard 分析發現的三個問題進行修正。**直接在 051 檔案上修改即可，不需要新建檔案。**

---

## 修正一：Position ISE 改為 per_step 版本

### 問題描述

目前 `control/Position_ISE` 是整集的積分值（`Σ cart_pos²`），episode 越長數值越大，導致訓練後期因為存活時間變長而數值膨脹，無法反映真實的車體漂移程度。

### 修改方式

在 `_compute_episode_metrics()` 中，新增 `Position_ISE_per_step` 指標：

```python
"Position_ISE_per_step": float(np.sum(positions ** 2) / n),
```

原本的 `Position_ISE`（整集積分值）保留不刪，兩個同時寫入 TensorBoard。

### TensorBoard 標籤

- 保留：`control/Position_ISE`
- 新增：`control/Position_ISE_per_step`

---

## 修正二：Settling Time 指標修復 + Position 獎勵加強

### 問題描述

Settling Time 有 96.8% 的 episode 都是 1，原因有二：
1. 初始擾動 ±0.1 弧度太小，大多數 episode 一開始就已在門檻內
2. 門檻 0.02 弧度太寬鬆

同時 Position ISE 後段暴增，代表 AI 犧牲車體位置換取角度穩定，`q_pos` 權重不足。

### 修改內容（共三處）

**修改 1：初始擾動範圍加大**

在 `AdvancedMotorWrapper.reset()` 中，將初始角度擾動從 `±0.1` 改為 `±0.2` 弧度：

```python
# 修改前
qpos[1] += np.random.uniform(-0.1, 0.1)

# 修改後
qpos[1] += np.random.uniform(-0.2, 0.2)
```

角速度擾動也同步加大：

```python
# 修改前
qvel[1] += np.random.uniform(-0.05, 0.05)

# 修改後
qvel[1] += np.random.uniform(-0.1, 0.1)
```

**修改 2：Settling Time 門檻收緊**

在 `step()` 中偵測穩定時間的判斷條件，將門檻從 `0.02` 改為 `0.01` 弧度（約 0.57 度）：

```python
# 修改前
if self.metric_settling_time is None and abs(pole_angle) < 0.02:

# 修改後
if self.metric_settling_time is None and abs(pole_angle) < 0.01:
```

**修改 3：提高位置懲罰權重**

在 `step()` 的 LQR 獎勵計算中，將 `q_pos` 的係數從 `1.0` 改為 `5.0`：

```python
# 修改前
q_pos = 1.0 * (cart_pos ** 2)

# 修改後
q_pos = 5.0 * (cart_pos ** 2)
```

同時更新穩態死區獎勵中的車體位置判定，將範圍從 `0.05` 放寬到 `0.1`，避免加強位置懲罰後死區獎勵太難觸發：

```python
# 修改前
if abs(cart_pos) < 0.05 and abs(cart_vel) < 0.05:

# 修改後
if abs(cart_pos) < 0.1 and abs(cart_vel) < 0.1:
```

---

## 修正三：調整單集最大步數 + 訓練結束時強制回傳最後一集數據

### 問題描述

目前 `max_episode_steps=100000`，導致訓練後半段 AI 每集存活數千到數萬步，50K 步之後 8 萬步的訓練區間只產生 7 個紀錄點。TensorBoard 後半段幾乎是空白，無法觀測訓練品質變化。

另外，訓練結束時如果最後一集尚未結束，callback 不會被觸發，導致最後一段訓練數據遺失。

### 修改內容（共兩處）

**修改 1：縮短單集最大步數**

在 `__main__` 中，將 `max_episode_steps` 從 `100000` 改為 `2000`：

```python
# 修改前
base_env = gym.make('InvertedPendulum-v4', max_episode_steps=100000)

# 修改後
base_env = gym.make('InvertedPendulum-v4', max_episode_steps=2000)
```

**修改 2：訓練結束時強制回收最後一集的指標**

在 `ControlMetricsCallback` 中新增 `_on_training_end()` 方法，訓練結束時主動從 env 取出未完成 episode 的指標並寫入 TensorBoard：

```python
class ControlMetricsCallback(BaseCallback):
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
                self.logger.dump(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        """
        🌟 訓練結束時，強制計算並寫入最後一集的控制指標。
        即使最後一集尚未因 terminated/truncated 結束，
        也把已累積的數據寫入 TensorBoard，避免數據遺失。
        """
        try:
            # 取得底層環境（穿透 VecEnv 包裝）
            env = self.training_env.envs[0]
            
            # 找到 AdvancedMotorWrapper 層
            wrapper = env
            while wrapper is not None:
                if hasattr(wrapper, '_compute_episode_metrics'):
                    break
                wrapper = getattr(wrapper, 'env', None)
            
            if wrapper is not None and wrapper.metric_step_count > 0:
                metrics = wrapper._compute_episode_metrics()
                self.episode_count += 1
                for key, value in metrics.items():
                    self.logger.record(f"control/{key}", value)
                self.logger.record("control/_final_episode", 1.0)
                self.logger.dump(self.num_timesteps)
                print(f"📊 已回收最後一集數據（{wrapper.metric_step_count} 步）寫入 TensorBoard")
        except Exception as e:
            print(f"⚠️ 回收最後一集數據時發生錯誤：{e}")
```

---

## 修改總覽

| 修改項目 | 檔案位置 | 修改前 | 修改後 |
|---|---|---|---|
| Position ISE per_step | `_compute_episode_metrics()` | 無此指標 | 新增 `Position_ISE_per_step` |
| 初始角度擾動 | `reset()` | `±0.1` 弧度 | `±0.2` 弧度 |
| 初始角速度擾動 | `reset()` | `±0.05` | `±0.1` |
| Settling Time 門檻 | `step()` 指標累積區 | `< 0.02` | `< 0.01` |
| q_pos 權重 | `step()` LQR 獎勵區 | `1.0` | `5.0` |
| 死區獎勵車位判定 | `step()` 穩態死區區 | `< 0.05` | `< 0.1` |
| max_episode_steps | `__main__` | `100000` | `2000` |
| 最後一集數據回收 | `ControlMetricsCallback` | 無此方法 | 新增 `_on_training_end()` |

---

## 預期效果

1. **Position ISE_per_step** 曲線不受 episode 長度影響，能清楚反映車體漂移的真實改善趨勢
2. **Settling Time** 出現有意義的收斂時間數據（不再全部是 1），能觀察 AI 從擾動恢復的速度
3. **Position ISE** 整體數值下降，AI 不再過度犧牲車體位置換取角度穩定
4. **TensorBoard 數據密度**：50K 步之後預期至少 25+ 個紀錄點（vs 修改前的 7 個），後半段曲線不再空白
5. **訓練結束時**確保最後一集的指標被寫入，不遺漏數據

---

## 輸出設定

### TensorBoard 紀錄路徑

```python
log_dir = "rl_project/tensorboard_logs/"
```

### TensorBoard run 名稱

```python
model.learn(
    total_timesteps=150000,
    tb_log_name="0513_Advanced_control",   # ← 使用此名稱
    callback=callback
)
```

### 模型儲存名稱

```python
model.save("0513")
print("✅ 訓練完成！模型已儲存為 0513.zip")
```

---

## 注意事項

- 所有修改都在原 051 檔案上進行，不需要新建檔案
- 保留所有未提及的功能不變（低通濾波、延遲、感測器雜訊、平滑度懲罰等）
- 中文註解與 emoji 風格保持一致
- 修改完成後請確認 `total_timesteps` 仍為 `150000`
- TensorBoard 紀錄存放於 `rl_project/tensorboard_logs/`，run 名稱為 `0513_Advanced_control`
- 訓練完成的模型儲存為 `0513.zip`

# 任務：半自動優化迴圈 — 從 0513 出發，自動訓練最多 10 輪

## 總覽

這是一個自動化訓練優化系統。Claude Code 需要：
1. 根據目前參數建立訓練腳本
2. 執行訓練
3. 解析 tfevents 分析本輪成效
4. 根據分析結果調整參數
5. 重複執行，直到達成停止條件或跑滿 10 輪

**全程不需要使用者確認，自動執行到底。**

---

## 目錄結構

```
rl_project/
├── 0513.zip                          ← 第一輪的起點（已存在，不可修改）
├── machinezip/
│   ├── 06111.zip                     ← 第一輪產出
│   ├── 06112.zip                     ← 第二輪產出
│   └── ...
└── tensorboard_machine_logs/
    ├── 06111_run_1/                  ← 第一輪 TensorBoard log
    ├── 06112_run_1/                  ← 第二輪 TensorBoard log
    └── ...
```

---

## 版本命名規則

版本名稱固定為以下 10 個，按順序使用：

```
06111, 06112, 06113, 06114, 06115,
06116, 06117, 06118, 06119, 06120
```

不使用 f-string 動態生成，直接用固定清單：

```python
versions = [
    "06111", "06112", "06113", "06114", "06115",
    "06116", "06117", "06118", "06119", "06120"
]
```

---

## 停止條件

每輪訓練結束後，解析該輪的 tfevents，計算：

```
short_episode_ratio = (Episode_Length < 50 的集次數) / 總集次數
```

若 `short_episode_ratio < 0.15`（即 < 50 步的集次低於 15%），**立即停止，不再執行下一輪**。

否則繼續下一輪，直到跑滿 10 輪為止。

---

## 每輪訓練腳本規格

### 觀測空間：5 維（與 0513 相容）

`AdvancedMotorWrapper` 的觀測空間必須是 5 維：
- 4 個狀態（cart_pos, pole_angle, cart_vel, pole_vel）
- 1 個馬達出力（last_real_action）

**不加入 delay_obs**，確保與 0513.zip 的網路輸入維度相容。

```python
new_shape = (orig_obs_space.shape[0] + self.action_space.shape[0],)  # 4+1=5
```

reset() 和 step() 的回傳：
```python
augmented_obs = np.concatenate((noisy_obs, self.last_real_action))
```

### delay=2 加權抽樣

在 `AdvancedMotorWrapper.__init__` 加入 `delay2_prob` 參數（預設 0.5），
在 `reset()` 裡用加權抽樣：

```python
if self.delay_randomize:
    self.delay_steps = np.random.choice(
        [1, 2], p=[1.0 - self.delay2_prob, self.delay2_prob]
    )
    self.action_queue = deque(maxlen=self.delay_steps + 1)
```

### 模型載入方式

```python
model = TD3.load(
    prev_zip,           # 第一輪為 "rl_project/0513.zip"，之後為上一輪的輸出
    env=env,
    verbose=1,
    tensorboard_log=log_dir,
)
model.action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions)
)
```

**重要**：`TD3.load()` 會從 zip 繼承 learning_rate、batch_size 等超參數，
不需要也不可以在 load 後重新指定這些值，直接呼叫 `model.learn()` 即可。

### 訓練與儲存

```python
os.makedirs("rl_project/machinezip", exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

model.learn(
    total_timesteps=150000,
    tb_log_name=version_name,
    callback=callback,
    reset_num_timesteps=True,
)
model.save(f"rl_project/machinezip/{version_name}")
```

---

## 可調參數與邊界

| 參數 | 初始值 | 可調下限 | 可調上限 |
|---|---|---|---|
| `disturbance_probability` | 0.3 | 0.2 | 0.5 |
| `disturbance_magnitude` 下限 | 0.1 | 0.05 | 0.2 |
| `disturbance_magnitude` 上限 | 0.3 | 0.2 | 0.5 |
| `disturbance_duration` 下限 | 2 | 1 | 3 |
| `disturbance_duration` 上限 | 5 | 3 | 8 |
| `delay2_prob` | 0.5 | 0.2 | 0.6 |
| `learning_rate` | 繼承自載入的 zip，**固定不調整，不可傳入** | — | — |

---

## 參數調整邏輯

每輪訓練結束後，解析 tfevents 計算：

```
short_ratio  = (Episode_Length < 50) / 總集次
dist_ratio   = (50 ≤ Episode_Length < 300) / 總集次
critic_trend = (後 25% critic_loss 均值) / (前 25% critic_loss 均值) - 1
               正值 = 上升（不穩定），負值 = 下降（收斂）
```

**規則按優先順序執行，每輪只套用第一個符合的規則：**

### 規則 D（最優先）：critic_trend > 0.5（Critic loss 上升超過 50%，訓練不穩定）
→ delay2_prob - 0.05（不低於 0.2）
→ disturbance_probability - 0.05（不低於 0.2）

### 規則 A：short_ratio > 0.4（delay=2 死亡嚴重）
→ delay2_prob - 0.05（不低於 0.2）

### 規則 C：dist_ratio > 0.25（擾動造成太多失敗）
→ disturbance_magnitude 上限 - 0.05（不低於 0.2）
→ disturbance_duration 上限 - 1（不低於 3）

### 規則 B：short_ratio ≤ 0.4 且 dist_ratio < 0.1（模型已能應對，可以加壓）
→ disturbance_probability + 0.05（不超過 0.5）

### 規則 E：以上均不符合
→ 所有參數不變，繼續下一輪

**注意**：規則 B 與規則 C 的觸發條件（0.1 ≤ dist_ratio ≤ 0.25）之間的區間會落入規則 E，這是設計意圖。

---

## tfevents 解析

使用純 Python 解析，不依賴 tensorboard 套件：

```python
import struct
from collections import defaultdict

def read_varint(data, pos):
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80): break
        shift += 7
    return result, pos

def read_field(data, pos):
    tag, pos = read_varint(data, pos)
    fn, wt = tag >> 3, tag & 0x07
    if wt == 0:
        val, pos = read_varint(data, pos); return fn, wt, val, pos
    elif wt == 1:
        val = struct.unpack_from('<d', data, pos)[0]; return fn, wt, val, pos+8
    elif wt == 2:
        l, pos = read_varint(data, pos); val = data[pos:pos+l]; return fn, wt, val, pos+l
    elif wt == 5:
        val = struct.unpack_from('<f', data, pos)[0]; return fn, wt, val, pos+4
    return fn, wt, None, pos

def parse_summary_value(data):
    pos, tag_str, simple_value = 0, None, None
    while pos < len(data):
        fn, wt, val, pos = read_field(data, pos)
        if fn == 1 and wt == 2: tag_str = val.decode('utf-8', errors='replace')
        elif fn == 2 and wt == 5: simple_value = val
    return tag_str, simple_value

def parse_summary(data):
    values, pos = [], 0
    while pos < len(data):
        fn, wt, val, pos = read_field(data, pos)
        if fn == 1 and wt == 2:
            t, v = parse_summary_value(val)
            if t and v is not None: values.append((t, v))
    return values

def parse_event(data):
    pos, wall_time, step, summary = 0, None, None, None
    while pos < len(data):
        fn, wt, val, pos = read_field(data, pos)
        if fn == 1 and wt == 1: wall_time = val
        elif fn == 2 and wt == 0: step = val
        elif fn == 5 and wt == 2: summary = val
    return wall_time, step, summary

def read_tb(filepath):
    """解析 tfevents 檔案，回傳 {tag: [(step, value), ...]}"""
    try:
        records = []
        with open(filepath, 'rb') as f:
            while True:
                lb = f.read(8)
                if len(lb) < 8: break
                l = struct.unpack('<Q', lb)[0]
                f.read(4); data = f.read(l); f.read(4)
                records.append(data)
        by_tag = defaultdict(list)
        for rec in records:
            wt, step, summary = parse_event(rec)
            if summary is not None and step is not None:
                for tag, val in parse_summary(summary):
                    by_tag[tag].append((step, val))
        for k in by_tag:
            by_tag[k] = sorted(by_tag[k], key=lambda x: x[0])
        return by_tag
    except Exception as e:
        print(f"⚠️ tfevents 解析失敗：{e}")
        return {}

def find_tfevents(log_dir):
    """遞迴尋找 log_dir 下的第一個 tfevents 檔案"""
    import os
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if 'tfevents' in f:
                return os.path.join(root, f)
    return None

def analyze_run(version_name, tb_log_base):
    """解析一輪訓練的 tfevents，回傳分析結果 dict"""
    import os, math
    log_dir = os.path.join(tb_log_base, f"{version_name}_run_1")
    tf_path = find_tfevents(log_dir)
    if tf_path is None:
        print(f"⚠️ 找不到 {log_dir} 的 tfevents，跳過分析")
        return None

    d = read_tb(tf_path)
    ep_len = d.get('control/Episode_Length', [])
    critic  = d.get('train/critic_loss', [])
    ise     = d.get('control/ISE_per_step', [])
    sse     = d.get('control/Steady_State_Error', [])

    if not ep_len:
        print("⚠️ 無 Episode_Length 資料，跳過分析")
        return None

    total = len(ep_len)
    vals  = [v for s, v in ep_len]
    short_ratio = sum(1 for v in vals if v < 50) / total
    dist_ratio  = sum(1 for v in vals if 50 <= v < 300) / total
    long_ratio  = sum(1 for v in vals if v >= 300) / total
    avg_len     = sum(vals) / total

    # Critic trend
    critic_vals = [v for s, v in critic if not math.isinf(v)]
    q = len(critic_vals) // 4
    if q >= 2:
        early_c = sum(critic_vals[:q]) / q
        late_c  = sum(critic_vals[-q:]) / q
        critic_trend = (late_c - early_c) / (abs(early_c) + 1e-9)
    else:
        critic_trend = 0.0

    ise_vals = [v for s, v in ise if not math.isinf(v)]
    sse_vals = [v for s, v in sse if not math.isinf(v)]
    ise_final = sum(ise_vals[-10:]) / min(len(ise_vals), 10) if ise_vals else 0
    sse_final = sum(sse_vals[-10:]) / min(len(sse_vals), 10) if sse_vals else 0

    return {
        "total": total,
        "avg_len": avg_len,
        "short_ratio": short_ratio,
        "dist_ratio": dist_ratio,
        "long_ratio": long_ratio,
        "critic_trend": critic_trend,
        "ise_final": ise_final,
        "sse_final": sse_final,
    }
```

---

## 每輪結束後的輸出報告格式

```
========================================
輪次：06111
訓練步數：150,000
起點模型：rl_project/0513.zip
輸出模型：rl_project/machinezip/06111.zip
----------------------------------------
📊 本輪分析：
  總集次：XXX
  平均集長：XXX 步
  short_ratio (< 50步)：XX.X%  ← 停止條件門檻 15%
  dist_ratio (50-300步)：XX.X%
  long_ratio (≥ 300步)：XX.X%
  critic_loss 趨勢：+XX.X%（上升）/ -XX.X%（下降）
  ISE_per_step 終值：X.XXXX
  Steady_State_Error 終值：X.XXXX
----------------------------------------
🔧 套用規則 X：
  delay2_prob：0.50 → 0.45
  disturbance_probability：0.30（不變）
  disturbance_magnitude：(0.10, 0.30)（不變）
  disturbance_duration：(2, 5)（不變）
----------------------------------------
⏭ 下一輪：06112，起點：rl_project/machinezip/06111.zip
[或] 🛑 停止條件達成（short_ratio=XX.X% < 15%），優化完成。
========================================
```

---

## 完整執行流程

```python
versions = [
    "06111", "06112", "06113", "06114", "06115",
    "06116", "06117", "06118", "06119", "06120"
]

TB_BASE = "rl_project/tensorboard_machine_logs"
params = {
    "delay2_prob": 0.5,
    "disturbance_probability": 0.3,
    "disturbance_magnitude": (0.1, 0.3),
    "disturbance_duration": (2, 5),
}
prev_zip = "rl_project/0513.zip"

for version in versions:
    log_dir = os.path.join(TB_BASE, f"{version}_run_1")
    os.makedirs("rl_project/machinezip", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 1. 建立環境與模型，執行訓練
    # （使用上方規格建立 env 和 model，呼叫 model.learn()）

    # 2. 解析 tfevents
    result = analyze_run(version, TB_BASE)
    if result is None:
        print(f"⚠️ {version} 分析失敗，強制停止")
        break

    # 3. 輸出報告
    # （依照報告格式印出）

    # 4. 判斷停止條件
    if result["short_ratio"] < 0.15:
        print(f"🛑 停止條件達成（short_ratio={result['short_ratio']*100:.1f}% < 15%）")
        break

    # 5. 套用參數調整規則（D → A → C → B → E）
    if result["critic_trend"] > 0.5:                          # 規則 D
        params["delay2_prob"]            = max(0.2, params["delay2_prob"] - 0.05)
        params["disturbance_probability"]= max(0.2, params["disturbance_probability"] - 0.05)
    elif result["short_ratio"] > 0.4:                         # 規則 A
        params["delay2_prob"] = max(0.2, params["delay2_prob"] - 0.05)
    elif result["dist_ratio"] > 0.25:                         # 規則 C
        lo, hi = params["disturbance_magnitude"]
        params["disturbance_magnitude"] = (lo, max(0.2, hi - 0.05))
        dlo, dhi = params["disturbance_duration"]
        params["disturbance_duration"]  = (dlo, max(3, dhi - 1))
    elif result["short_ratio"] <= 0.4 and result["dist_ratio"] < 0.1:  # 規則 B
        params["disturbance_probability"] = min(0.5, params["disturbance_probability"] + 0.05)
    # 規則 E：不變

    # 6. 更新起點
    prev_zip = f"rl_project/machinezip/{version}.zip"
```

---

## 不可更動的部分

以下在所有輪次中固定不變：

- `AdvancedMotorWrapper` 的完整獎勵函數（LQR 懲罰、超級獎勵、clip）
- `ControlMetricsCallback` 的完整實作（包含 `_on_training_end` 回收最後一集）
- `max_episode_steps=2000`
- `action_noise` 的 `sigma=0.2`
- `learning_rate`、`policy_delay`、`target_policy_noise`、`batch_size`、`buffer_size` 全部繼承自載入的 zip，不重新指定
- 每輪訓練步數 150,000

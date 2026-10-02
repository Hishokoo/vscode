# CLAUDE.md - rl_project

> **Documentation Version**: 1.0
> **Last Updated**: 2026-05-19
> **Project**: rl_project
> **Description**: MuJoCo 強化學習實驗專案（DDPG / Stable-Baselines3）
> **Features**: GitHub auto-backup, Task agents, technical debt prevention

This file provides essential guidance to Claude Code when working with code in this repository.

## CRITICAL RULES - READ FIRST

### RULE ACKNOWLEDGMENT REQUIRED
Before starting ANY task, respond with:
"CRITICAL RULES ACKNOWLEDGED - I will follow all prohibitions and requirements listed in CLAUDE.md"

### ABSOLUTE PROHIBITIONS
- **NEVER** create duplicate script files (train_v2.py, enhanced_ddpg.py) — extend existing files
- **NEVER** hardcode paths — use the path conventions defined below
- **NEVER** use naming like `enhanced_`, `improved_`, `new_`, `v2_` — extend original files
- **NEVER** use git commands with -i flag
- **NEVER** create multiple implementations of the same concept
- **NEVER** copy-paste code blocks — extract into `tools/` as shared utilities
- **NEVER** write output files (models, logs) to root — use designated folders

### MANDATORY REQUIREMENTS
- **COMMIT** after every completed experiment or code change
- **GITHUB BACKUP** — push after every commit: `git push origin main`
- **USE TASK AGENTS** for training runs (>30 seconds)
- **TODOWRITE** for multi-step tasks (3+ steps)
- **READ FILES FIRST** before editing any script
- **DEBT PREVENTION** — search for existing utilities in `tools/` before creating new ones

---

## PROJECT STRUCTURE

```
rl_project/
├── CLAUDE.md
├── .vscode/
│   └── settings.json          # conda 環境設定
├── tools/                     # 共用工具腳本
│   ├── run_training.sh        # 統一訓練啟動腳本
│   └── plot_results.py        # TensorBoard / 結果視覺化
├── prompts/                   # 從 Claude.ai 製作的提示詞存放區
├── output/                    # 所有輸出（模型、圖表、報告）統一放這
│   └── models/                # 訓練好的 .zip 模型（唯一存放位置，不放在腳本旁）
├── tensorboard_logs/          # TensorBoard 紀錄，依實驗分類存放（子資料夾對應 experiments/ 分類）
│   ├── ddpg/                  # 011~041
│   ├── td3/                   # 05111~05114
│   ├── evaluation/             # 05121 壓力測試
│   ├── curriculum/            # 06111~06120
│   └── machine/                # 07111 起
│
└── experiments/                       # 訓練/評估腳本，依實驗階段分類存放
    ├── ddpg/                          # 011~043：DDPG 基礎系列
    │   ├── 011train_mujoco_ddpg.py    # 實驗 011：基礎 DDPG
    │   ├── 012testmojoco.py
    │   ├── 021train_mujoco_ddpg.py    # 實驗 021：加入真實馬達噪音
    │   ├── 022testmojoco.py
    │   ├── 023testmojoco.py
    │   ├── 031trytrain_mujoco_ddpg.py # 實驗 031：靜態穩定
    │   ├── 032trytrain_mujoco_ddpg.py
    │   ├── 041trytrain_mujoco_ddpg.py # 實驗 041：進階控制
    │   ├── 042trytrain_mujoco_ddpg.py
    │   └── 043trytrain_mujoco_ddpg.py
    ├── td3/                           # 05111~05114：TD3 主線
    │   ├── 05111trytrain_mujoco_td3.py
    │   ├── 05112trytrain_mujoco_td3.py
    │   ├── 05113trytrain_mujoco_td3.py
    │   └── 05114trytrain_mujoco_td3.py
    ├── evaluation/                    # 05121：純評估，不訓練
    │   └── 05121stress_test.py        # 依 importlib 從 ../td3/ 動態載入 Wrapper
    ├── curriculum/                    # 06111~06120：課程學習 + 自動調參迴圈
    │   ├── 06111trytrain_mujoco_td3.py
    │   └── 0611auto_optimize_loop.py
    └── machine/                       # 07111 起：Sim-to-Real 領域隨機化重校
        └── 07111trytrain_mujoco_td3.py
```

**執行慣例（務必遵守，否則相對路徑會解析錯誤）**：所有訓練/評估腳本一律以 `rl_project/` 為工作目錄執行，例如：

```bash
cd rl_project
python experiments/td3/05114trytrain_mujoco_td3.py
```

腳本內的模型存讀路徑（`output/models/xxx`）都是相對於這個工作目錄，與腳本自己被搬到哪個子資料夾無關。`05121stress_test.py` 是例外——它用 `__file__` 回推到 `rl_project/` 根目錄，因此不論從哪個工作目錄執行都能正確解析。

---

## NAMING CONVENTION

### 腳本編號規則
```
[實驗組][序號][功能].py
 ↑        ↑     ↑
 01~09   1~9  train / test / try
```

| 前綴 | 意義 |
|------|------|
| `XXXtrain_` | 主訓練腳本 |
| `XXXtest_`  | 測試 / 推論腳本 |
| `XXXtry_`   | 實驗性修改 |

### TensorBoard log 名稱規則
```python
tb_log_name = "0XX_描述"   # 對應實驗組號，例如 "021_Realistic_Motor"
tensorboard_log = "tensorboard_logs/<分類>/"   # 分類對應 experiments/ 底下的子資料夾
                                                # 例如 ddpg / td3 / evaluation / curriculum / machine
```

---

## COMMON COMMANDS

```bash
# 啟動訓練（範例；務必在 rl_project/ 目錄下執行）
python experiments/ddpg/011train_mujoco_ddpg.py
python experiments/td3/05114trytrain_mujoco_td3.py
python experiments/evaluation/05121stress_test.py

# 查看 TensorBoard
tensorboard --logdir tensorboard_logs/

# 推送備份
git add -A && git commit -m "exp: 描述" && git push origin main
```

---

## ENVIRONMENT

- **Framework**: Stable-Baselines3
- **Simulator**: MuJoCo (gymnasium)
- **Algorithm**: DDPG（Deep Deterministic Policy Gradient）
- **Python env**: conda（見 .vscode/settings.json）

---

## TECHNICAL DEBT PREVENTION

### WRONG:
```python
# 直接在腳本裡複製整段訓練邏輯
# 建立 train_v2.py 而不是修改原檔
```

### CORRECT:
```python
# 1. 先在 tools/ 找有沒有現成工具
# 2. 讀現有腳本再決定要改哪裡
# 3. 擴充原腳本，不新增重複檔案
```

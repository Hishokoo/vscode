# CLAUDE.md - RL_Lab

> **Documentation Version**: 1.0
> **Last Updated**: 2026-05-19
> **Project**: RL_Lab
> **Description**: Gymnasium 強化學習入門實驗室（CartPole、Classic Control）
> **Features**: GitHub auto-backup, Task agents, technical debt prevention

This file provides essential guidance to Claude Code when working with code in this repository.

## CRITICAL RULES - READ FIRST

### RULE ACKNOWLEDGMENT REQUIRED
Before starting ANY task, respond with:
"CRITICAL RULES ACKNOWLEDGED - I will follow all prohibitions and requirements listed in CLAUDE.md"

### ABSOLUTE PROHIBITIONS
- **NEVER** create duplicate script files (lab_v2.py, enhanced_cartpole.py) — extend existing files
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
- **DEBT PREVENTION** — search for existing utilities before creating new ones

---

## PROJECT STRUCTURE

```
RL_Lab/
├── CLAUDE.md
├── .vscode/
│   └── settings.json
├── tools/                     # 共用工具腳本
├── output/                    # 所有輸出（模型、圖表）統一放這
│
├── 01RL_Lab.py                # 實驗 01：CartPole 隨機動作入門
```

---

## NAMING CONVENTION

### 腳本編號規則
```
[序號][功能].py
 ↑     ↑
01~99  描述性名稱
```

| 前綴 | 意義 |
|------|------|
| `XXtrain_` | 主訓練腳本 |
| `XXtest_`  | 測試 / 推論腳本 |
| `XXlab_`   | 實驗性探索 |

---

## COMMON COMMANDS

```bash
# 執行實驗
python 01RL_Lab.py

# 安裝依賴
pip install "gymnasium[classic-control]"

# 推送備份
git add -A && git commit -m "exp: 描述" && git push origin main
```

---

## ENVIRONMENT

- **Framework**: Gymnasium
- **Environments**: Classic Control（CartPole-v1 等）
- **Algorithm**: 從隨機動作 → DQN → Policy Gradient 逐步進階
- **Python env**: conda（見 .vscode/settings.json）

---

## TECHNICAL DEBT PREVENTION

### WRONG:
```python
# 直接複製 01RL_Lab.py 並改名為 01RL_Lab_v2.py
# 在根目錄建立輸出檔案
```

### CORRECT:
```python
# 1. 先在 tools/ 找有沒有現成工具
# 2. 讀現有腳本再決定要改哪裡
# 3. 擴充原腳本，不新增重複檔案
# 4. 輸出一律放 output/
```

#!/bin/bash
# 統一訓練啟動腳本
# 用法: bash tools/run_training.sh 011train_mujoco_ddpg.py

SCRIPT=$1

if [ -z "$SCRIPT" ]; then
    echo "用法: bash tools/run_training.sh <腳本名稱>"
    echo "範例: bash tools/run_training.sh 011train_mujoco_ddpg.py"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "找不到腳本: $SCRIPT"
    exit 1
fi

echo "=============================="
echo "開始訓練: $SCRIPT"
echo "時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================="

python "$SCRIPT"

echo "=============================="
echo "訓練結束: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================="

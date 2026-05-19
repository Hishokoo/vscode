"""
查看所有實驗的 TensorBoard log 清單與基本統計
用法: python tools/plot_results.py
"""
import os
from pathlib import Path

LOG_DIR = Path("tensorboard_logs")


def list_experiments():
    if not LOG_DIR.exists():
        print("找不到 tensorboard_logs/ 資料夾")
        return

    experiments = sorted([d for d in LOG_DIR.iterdir() if d.is_dir()])
    if not experiments:
        print("目前沒有任何實驗紀錄")
        return

    print(f"{'實驗名稱':<35} {'大小':>10}")
    print("-" * 50)
    for exp in experiments:
        size = sum(f.stat().st_size for f in exp.rglob("*") if f.is_file())
        size_kb = size / 1024
        print(f"{exp.name:<35} {size_kb:>8.1f} KB")

    print()
    print(f"共 {len(experiments)} 個實驗")
    print(f"\n啟動 TensorBoard: tensorboard --logdir {LOG_DIR}")


if __name__ == "__main__":
    list_experiments()

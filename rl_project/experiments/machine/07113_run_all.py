# 07113_run_all.py
# 依 prompts/07113_pure_rl_changes.md 第 8 節的實驗計畫產生並執行 07113 訓練指令（每個 seed 一個獨立程序）。
#
#   E0      --preset 07112                     07112 在新評估下的真實水準（基準線）
#   E1      新雜訊參數（預設值）                探索雜訊是不是主因
#   E2      E1 + --ctrl-hz 25                  時間尺度是不是主因
#   E3a/b/c E1 或 E2（--e3-base）+ CAPS λ = 0.1 / 0.3 / 1.0
#   E4      E3 中最佳（--e4-from）+ --asym-critic
#
#   python experiments/machine/07113_run_all.py --only E0 E1 E2 --jobs 6
#   python experiments/machine/07113_run_all.py --only E3a E3b E3c --e3-base E2
#   python experiments/machine/07113_run_all.py --only E4 --e3-base E2 --e4-from E3b
#   python experiments/machine/07113_run_all.py --smoke --jobs 8      # 冒煙測試（tag 加 -smoke，彙整時預設排除）
#   python experiments/machine/07113_run_all.py --dry-run             # 只列印指令
#
# 每個程序的輸出寫到 output/logs/07113/{tag}_s{seed}.log。

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]                          # rl_project/
TRAIN_SCRIPT = HERE / "07113trytrain_mujoco_td3.py"
LOG_DIR = PROJECT_ROOT / "output" / "logs" / "07113"

CODES = ["E0", "E1", "E2", "E3a", "E3b", "E3c", "E4"]
CAPS_LAMBDA = {"E3a": 0.1, "E3b": 0.3, "E3c": 1.0}
SMOKE_ARGS = ["--timesteps", "20000", "--eval-freq", "10000", "--eval-episodes", "2",
              "--final-eval-episodes", "2"]


def build_experiments(e3_base, e4_from):
    exps = {"E0": ["--preset", "07112"], "E1": [], "E2": ["--ctrl-hz", "25"]}
    for code, lam in CAPS_LAMBDA.items():
        exps[code] = exps[e3_base] + ["--caps-lambda-t", str(lam), "--caps-lambda-s", str(lam)]
    exps["E4"] = exps[e4_from] + ["--asym-critic"]
    return exps


def run_one(cmd, log_path, env):
    with open(log_path, "w", encoding="utf-8") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT, env=env).returncode
    print(f"{'✅' if rc == 0 else '❌'} rc={rc}  {log_path.name}", flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=CODES, default=CODES, help="只跑指定代號")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=3, help="同時執行的程序數（注意 CPU 核心數）")
    ap.add_argument("--e3-base", choices=["E1", "E2"], default="E1", help="E3 疊加在 E1 或 E2 上")
    ap.add_argument("--e4-from", choices=list(CAPS_LAMBDA), default="E3b", help="E4 以哪個 E3 為底")
    ap.add_argument("--smoke", action="store_true", help="冒煙測試：20K 步、評估集數 2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="-- 之後的參數原樣傳給訓練腳本")
    args = ap.parse_args()
    extra = [a for a in args.extra if a != "--"]

    exps = build_experiments(args.e3_base, args.e4_from)
    jobs = []
    for code in args.only:
        tag = f"{code}-smoke" if args.smoke else code
        for seed in args.seeds:
            cmd = ([sys.executable, str(TRAIN_SCRIPT), "--tag", tag, "--seed", str(seed)]
                   + exps[code] + (SMOKE_ARGS if args.smoke else []) + extra)
            jobs.append((cmd, LOG_DIR / f"{tag}_s{seed}.log"))

    for cmd, log in jobs:
        print(" ".join(cmd[1:]), "→", log.name)
    if args.dry_run:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    threads = str(max(1, (os.cpu_count() or 1) // max(1, args.jobs)))
    env.setdefault("OMP_NUM_THREADS", threads)     # 避免多個程序搶同一批核心
    env.setdefault("MKL_NUM_THREADS", threads)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rcs = list(pool.map(lambda j: run_one(j[0], j[1], env), jobs))
    n_fail = sum(rc != 0 for rc in rcs)
    print(f"\n完成 {len(rcs) - n_fail}/{len(rcs)}；彙整：python experiments/machine/07113_aggregate.py"
          + (" --include-smoke" if args.smoke else ""))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

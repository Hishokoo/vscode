# 07113_aggregate.py
# 彙整 07113 多 seed 的最終評估：讀 output/results/07113_*_final_eval.json，依設定標籤（setting）分組，
# 每組列出各指標「跨 seed 平均 ± 標準差」與對同頻率 LQI 的倍數（每個 seed 先算 RL/LQI 再取平均 ± 標準差）。
# 輸出 markdown 到 stdout，並另存 output/results/07113_summary.md。
#
#   python experiments/machine/07113_aggregate.py                 # 預設排除冒煙測試（tag 含 "smoke"）
#   python experiments/machine/07113_aggregate.py --include-smoke

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # rl_project/
RESULT_DIR = PROJECT_ROOT / "output" / "results"

METRICS = ["full_length_frac", "return_mean", "osc_x_std_tail", "osc_theta_std_tail",
           "osc_theta_zc_hz_tail", "max_cart_pos_abs", "RMS_Control_Effort",
           "Duty_Saturation_frac", "rail_limit_hit", "angle_limit_hit"]
POLICIES = ["rl_final", "rl_best"]
ENVS = ["nominal", "dr"]


def _ms(vals, fmt="{:.4g}"):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "—"
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(np.mean(vals))} ± {fmt.format(np.std(vals, ddof=1))}"


def _ratio(rl, lqi):
    if rl is None or lqi is None or abs(lqi) < 1e-12:
        return None
    return rl / lqi


def _get(run, policy, env, metric):
    return run["summary"].get(f"{policy}_{env}", {}).get(metric)


def load_runs(include_smoke):
    groups = defaultdict(list)
    for path in sorted(RESULT_DIR.glob("07113_*_final_eval.json")):
        with open(path, encoding="utf-8") as f:
            run = json.load(f)
        if not include_smoke and "smoke" in run["tag"]:
            continue
        groups[run["setting"]].append(run)
    return groups


def overview_table(groups):
    """決策用總表：名目 x 標準差倍數（目標 ≤ 2）與 angle_limit_hit。"""
    lines = ["| setting | seeds | 名目 x_std / LQI（final） | 名目 x_std / LQI（best） | "
             "名目 angle_hit（final） | DR full_len（final） | DR angle_hit（final） |",
             "|---|---|---|---|---|---|---|"]
    for setting, runs in sorted(groups.items()):
        r = lambda pol: [_ratio(_get(x, pol, "nominal", "osc_x_std_tail"),
                                _get(x, "lqi", "nominal", "osc_x_std_tail")) for x in runs]
        g = lambda pol, env, m: [_get(x, pol, env, m) for x in runs]
        lines.append(f"| {setting} | {', '.join(str(x['seed']) for x in runs)} | "
                     f"{_ms(r('rl_final'), '{:.2f}')}× | {_ms(r('rl_best'), '{:.2f}')}× | "
                     f"{_ms(g('rl_final', 'nominal', 'angle_limit_hit'), '{:.3f}')} | "
                     f"{_ms(g('rl_final', 'dr', 'full_length_frac'), '{:.3f}')} | "
                     f"{_ms(g('rl_final', 'dr', 'angle_limit_hit'), '{:.3f}')} |")
    return lines


def setting_tables(setting, runs):
    a = runs[0]
    lines = [f"## {setting}", "",
             f"seeds = {[x['seed'] for x in runs]}，algo = {a['algo']}，"
             f"ctrl_dt = {a['timing']['ctrl_dt']}，gamma = {a['timing']['gamma']}，"
             f"最終評估種子 {a['final_eval_seeds'][0]}–{a['final_eval_seeds'][1]}", ""]
    for env in ENVS:
        lines += [f"### {env}", "",
                  "| 指標 | RL final | RL best | LQI | final / LQI | best / LQI |",
                  "|---|---|---|---|---|---|"]
        for m in METRICS:
            lqi = [_get(x, "lqi", env, m) for x in runs]
            cells = [_ms([_get(x, p, env, m) for x in runs]) for p in POLICIES]
            ratios = [_ms([_ratio(_get(x, p, env, m), l) for x, l in zip(runs, lqi)], "{:.2f}")
                      for p in POLICIES]
            lines.append(f"| {m} | {cells[0]} | {cells[1]} | {_ms(lqi)} | {ratios[0]} | {ratios[1]} |")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-smoke", action="store_true", help="納入 tag 含 smoke 的冒煙測試結果")
    args = ap.parse_args()

    groups = load_runs(args.include_smoke)
    if not groups:
        print(f"找不到結果檔：{RESULT_DIR}/07113_*_final_eval.json")
        return
    lines = ["# 07113 彙整（跨 seed 平均 ± 標準差；倍數 = 每個 seed 的 RL / 同頻率 LQI）", "",
             "成功判準：名目 osc_x_std_tail 倍數 ≤ 2，且 angle_limit_hit 不高於 07112（E0）。"
             "進步需大於跨 seed 標準差才算數。", "",
             "## 總表", ""] + overview_table(groups) + [""]
    for setting, runs in sorted(groups.items()):
        lines += setting_tables(setting, runs)
    text = "\n".join(lines)
    print(text)
    out = RESULT_DIR / "07113_summary.md"
    out.write_text(text + "\n", encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()

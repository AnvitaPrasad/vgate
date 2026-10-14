"""Apply the preregistered Gate-1 criteria (experiments/PREREG.md) to an analysis JSON -> markdown table.

  python scripts/gate1_report.py experiments/runs/J1/analysis__Qwen3-VL-4B-Instruct__dev.json [--c2 path_to_c2.json]
Numbers are read, never re-derived by hand. Criteria text mirrors PREREG v0.
"""
import json
import sys

import numpy as np


def main():
    a = json.load(open(sys.argv[1]))
    ds = [k for k in a if k != "pooled_dnauc"]
    rows = []
    # C1-a headroom: always - never >= 5 pt on >= 1 dataset AND help >= 8%
    ha = {d: (a[d]["quadrants"]["acc_always"][0] - a[d]["quadrants"]["acc_never"][0]) * 100 for d in ds}
    help_ = {d: a[d]["quadrants"]["help"][0] * 100 for d in ds}
    ok_a = any(ha[d] >= 5 and help_[d] >= 8 for d in ds)
    rows.append(("C1-a headroom", ", ".join(f"{d}: +{ha[d]:.1f} pt, help {help_[d]:.1f}%" for d in ds), ok_a))
    # C1-b harm pooled >= 2.5%
    harm = np.mean([a[d]["quadrants"]["harm"][0] for d in ds]) * 100
    rows.append(("C1-b pooled harm >= 2.5%", f"{harm:.2f}%", harm >= 2.5))
    # C1-c label effect (same features): dnAUC >= +0.75 pt with CI > 0, per feature set
    for fs in ["pre", "post"]:
        k = f"benefit_{fs} - error_{fs}"
        if k in a["pooled_dnauc"]:
            pt, lo, hi = (100 * x for x in a["pooled_dnauc"][k])
            rows.append((f"C1-c label effect ({fs}-search features)", f"{pt:+.2f} pt [{lo:+.2f}, {hi:+.2f}]",
                         pt >= 0.75 and lo > 0))
    # C1-d practical: some gate within 0.5 pt of always at <= 0.75x total tokens on >= 1 dataset
    best = {d: min((v["rel_cost_at_always_acc"], k) for k, v in a[d]["policies"].items()
                   if isinstance(v, dict) and "rel_cost_at_always_acc" in v and k != "oracle"
                   and not np.isnan(v["rel_cost_at_always_acc"])) for d in ds}
    rows.append(("C1-d total-token cost <= 0.75x at always-acc - 0.5pt",
                 ", ".join(f"{d}: {best[d][0]:.2f}x ({best[d][1]})" for d in ds), any(best[d][0] <= 0.75 for d in ds)))
    # Reframing evidence (not a preregistered criterion): search calls needed
    srch = {d: min((v.get("search_rate_at_always_acc", np.nan), k) for k, v in a[d]["policies"].items()
                   if isinstance(v, dict) and "search_rate_at_always_acc" in v and k != "oracle"
                   and not np.isnan(v.get("search_rate_at_always_acc", np.nan))) for d in ds}
    rows.append(("(descriptive) searches needed for always-acc - 0.5pt",
                 ", ".join(f"{d}: {srch[d][0]:.0%} ({srch[d][1]})" for d in ds), None))
    print("| criterion | value | pass |\n|---|---|---|")
    for n, v, ok in rows:
        print(f"| {n} | {v} | {'—' if ok is None else ('PASS' if ok else 'FAIL')} |")


if __name__ == "__main__":
    main()

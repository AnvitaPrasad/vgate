"""Holm correction over the family of PREREGISTERED tests (the exploratory analyses are not part of it).

Each test is a paired difference with a stored 95% bootstrap interval (confirm.py / extend.py). The two-sided p-value
uses a normal approximation to that interval: SE = (hi - lo) / (2 * 1.96), z = point / SE. Degenerate intervals
(lo == hi == point, e.g. a gate that collapses onto a fixed policy) get p = 1.

  python scripts/multiplicity.py   -> experiments/results/multiplicity.{csv,md}

Family (see experiments/PREREG.md): v1 primary on models 1-2; primary and probe on models 3-4 (declared before their
outputs existed); on OK-VQA (v2 / v2b) primary, probe, VGate-full vs VGate-post and vs VGate-pre for both Qwen models,
H-OK (the gate at its dev-chosen read rate beats never-retrieve; it must also beat always-retrieve) and H-SEL.
The hidden-state probe comparison on models 1-2 for E-VQA / InfoSeek was scored in a second pass and is NOT in the family.
"""
import json
import math
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "experiments/runs"
M1, M2, M3, M4 = "Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct", "InternVL3_5-4B-HF", "Qwen3-VL-8B-Instruct"
ALPHA = 0.05


def p_from_ci(v):
    pt, lo, hi = v
    if hi - lo < 1e-12:
        return 1.0 if abs(pt) < 1e-12 else 0.0
    z = abs(pt) / ((hi - lo) / (2 * 1.959964))
    return math.erfc(z / math.sqrt(2))


def main():
    J = lambda run, kind, m: json.load(open(R / run / f"{kind}__{m}.json"))
    tests = []

    def add(name, prereg, v):
        tests.append(dict(test=name, prereg=prereg, point=100 * v[0], lo=100 * v[1], hi=100 * v[2], p=p_from_ci(v)))
    P, B = "benefit_post - error_post", "probe_benefit - probe_error"
    add("primary, Qwen3-VL-4B, E-VQA+InfoSeek", "v1", J("J4", "confirm", M1)["pooled_dnauc_95ci"][P])
    add("primary, Qwen2.5-VL-3B, E-VQA+InfoSeek", "v1 (replication)", J("J2", "confirm", M2)["pooled_dnauc_95ci"][P])
    for run, m, tag in [("J5", M3, "2026-10-03"), ("J7", M4, "v2c")]:
        c = J(run, "confirm", m)["pooled_dnauc_95ci"]
        add(f"primary, {m.replace('-Instruct', '').replace('_', '.').replace('-HF', '')}, E-VQA+InfoSeek", tag, c[P])
        add(f"probe, {m.replace('-Instruct', '').replace('_', '.').replace('-HF', '')}, E-VQA+InfoSeek", tag, c[B])
    for m in (M1, M2):
        c, e = J("J6", "confirm", m)["pooled_dnauc_95ci"], J("J6", "extend", m)
        short = m.replace("-Instruct", "")
        add(f"primary, {short}, OK-VQA", "v2", c[P]); add(f"probe, {short}, OK-VQA", "v2", c[B])
        add(f"VGate-full - VGate-post, {short}, OK-VQA", "v2b", e["pooled_dnauc_95ci"]["vgate_full - benefit_post"])
        add(f"VGate-full - VGate-pre, {short}, OK-VQA", "v2b", e["pooled_dnauc_95ci"]["vgate_full - probe_benefit"])
        add(f"H-OK (gate - never at dev-chosen rate), {short}, OK-VQA", "v2", e["okvqa"]["hok"]["minus_never_95ci"])
        add(f"H-SEL (select - always), {short}, OK-VQA", "v2b", e["okvqa"]["select_minus_always_95ci"])
    for run, m in (("J11", M3), ("J12", M4)):  # PREREG v4c: the OK-VQA endpoints of v2b on models 3 and 4 (added 2026-10-06)
        if (R / run / f"confirm__{m}.json").exists():
            c = J(run, "confirm", m)["pooled_dnauc_95ci"]
            short = m.replace("-Instruct", "").replace("_", ".").replace("-HF", "")
            add(f"primary, {short}, OK-VQA", "v4c", c[P]); add(f"probe, {short}, OK-VQA", "v4c", c[B])
    df = pd.DataFrame(tests).sort_values("p").reset_index(drop=True)
    m, alive = len(df), True
    df["holm_threshold"] = [ALPHA / (m - i) for i in range(m)]
    keep = []
    for i, row in df.iterrows():  # step-down: stop at the first failure
        alive = alive and row.p <= row.holm_threshold
        keep.append(alive)
    df["significant_after_holm"] = keep
    df["unadjusted_ci_excludes_0"] = (df.lo > 0) | (df.hi < 0)
    out = ROOT / "experiments/results/multiplicity"
    df.round(6).to_csv(str(out) + ".csv", index=False)
    lines = [f"# Holm correction over the {m} preregistered tests (alpha = {ALPHA}, two-sided, normal approximation to the bootstrap intervals)\n",
             f"Significant after correction: **{int(df.significant_after_holm.sum())} of {m}** "
             f"(unadjusted intervals excluding 0: {int(df.unadjusted_ci_excludes_0.sum())}).\n",
             "| test | prereg | point [95% CI] | p | Holm threshold | significant |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.test} | {r.prereg} | {r.point:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}] | {r.p:.2g} | {r.holm_threshold:.4f} | "
                     f"{'yes' if r.significant_after_holm else 'no'} |")
    lines.append("\nH-SEL is significant only in the trivial sense recorded in the outcome log (selection recovers never-retrieve).")
    pathlib.Path(str(out) + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""PREREG v12 (EXPLORATORY): the planner's noise floor against the case study's cells and the published log.

For each benchmark-reader cell of experiments/results/gate_bound.json: the exact null lead (vgate.gate.null_lead: what the
best threshold of a score with no signal leads the better fixed policy by), the same with the sign of the score chosen too
(shuffles), and the mean best lead of that cell's scores (threshold and sign chosen on the held-out labels) and their mean
split-half lead. Then the same floor for the published log of PREREG v11 (GPT-4o; correct above 0.5 token accuracy) and for
two reference logs.

  python scripts/null_lead_check.py   -> experiments/results/null_lead.{json,md}
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vgate import gate  # noqa: E402


def with_sign(h, m, n, reps=4000, seed=0):
    """Null lead when the threshold and the direction of the score are both chosen on the labels (shuffled orders)."""
    P, Q = int(round(h * n)), int(round(m * n))
    if P == 0 or Q == 0:
        return 0.0
    rng, u, out = np.random.default_rng(seed), np.array([1] * P + [-1] * Q), []
    for _ in range(reps):
        c = np.concatenate([[0], np.cumsum(rng.permutation(u))])
        out.append(max(c.max(), (c[-1] - c).max()) - max(0, c[-1]))
    return float(np.mean(out) / n)


def main():
    G = json.load(open(ROOT / "experiments/results/gate_bound.json"))
    pts = pd.DataFrame(G["points"])
    rep, rows = dict(cells=[], logs=[]), []
    for c in G["cells"]:
        h, m, n = c["help"] / 100, c["harm"] / 100, c["n"]
        k = pts[(pts.model == c["model"]) & (pts.benchmark == c["benchmark"])]
        rep["cells"].append(dict(model=c["model"], benchmark=c["benchmark"], n=n, help=c["help"], harm=c["harm"],
                                 null_lead=round(100 * gate.null_lead(h, m, n), 3), null_lead_with_sign=round(100 * with_sign(h, m, n), 3),
                                 scores=len(k), mean_best_lead=round(float(k.observed.mean()), 3), mean_split_half_lead=round(float(k.split.mean()), 3),
                                 scores_above_null_with_sign=int((k.observed > 100 * with_sign(h, m, n)).sum())))
    C = pd.DataFrame(rep["cells"])
    rep["summary"] = dict(null_lead_max=float(C.null_lead.max()), null_lead_with_sign_max=float(C.null_lead_with_sign.max()),
                          null_lead_with_sign_mean=round(float((C.null_lead_with_sign * C.scores).sum() / C.scores.sum()), 3),
                          mean_best_lead=round(float(pts.observed.mean()), 3), mean_split_half_lead=round(float(pts.split.mean()), 3),
                          scores_above_null_with_sign=int(C.scores_above_null_with_sign.sum()), scores=int(C.scores.sum()))
    d = pd.read_csv(ROOT / "demo/sample_log_published.csv"); u = (d.c_ev - d.c_ne).to_numpy()
    logs = [("published log (GPT-4o, six sources pooled)", float((u == 1).mean()), float((u == -1).mean()), len(u))]
    logs += [(f"published log, source {s}", float((u[d.segment == s] == 1).mean()), float((u[d.segment == s] == -1).mean()), int((d.segment == s).sum())) for s in d.segment.unique()]
    logs += [("reference: helps 10%, harms 10%, 1,000 queries", 0.10, 0.10, 1000), ("reference: helps 10%, harms 10%, 10,000 queries", 0.10, 0.10, 10000)]
    for name, h, m, n in logs:
        rep["logs"].append(dict(log=name, n=n, help=round(100 * h, 1), harm=round(100 * m, 1), null_lead=round(100 * gate.null_lead(h, m, n), 2),
                                null_lead_with_sign=round(100 * with_sign(h, m, n), 2), oracle_lead=round(100 * min(h, m), 1)))
    S = rep["summary"]
    L = ["# The planner's noise floor (PREREG v12; exploratory)", "",
         "Null lead: what the best threshold of a score with no signal leads the better fixed policy by (points), exactly; with sign: the direction of the score chosen too (4,000 shuffles).", "",
         f"Case study, {S['scores']} held-out scores in {len(C)} cells: the null lead is at most {S['null_lead_max']:.3f} points ({S['null_lead_with_sign_max']:.3f} with sign; "
         f"{S['null_lead_with_sign_mean']:.3f} on average over the scores). The scores' best leads average {S['mean_best_lead']:.2f} points and their split-half leads "
         f"{S['mean_split_half_lead']:.2f}; {S['scores_above_null_with_sign']} of {S['scores']} best leads are above their cell's null lead with sign.", "",
         "| reader | benchmark | n | helps % | harms % | null lead | with sign | mean best lead of its scores | mean split-half lead |", "|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {r.model} | {r.benchmark} | {r.n} | {r.help} | {r.harm} | {r.null_lead} | {r.null_lead_with_sign} | {r.mean_best_lead} | {r.mean_split_half_lead} |" for r in C.itertuples()]
    L += ["", "| log | n | helps % | harms % | oracle lead | null lead | with sign |", "|---|---|---|---|---|---|---|"]
    L += [f"| {r['log']} | {r['n']} | {r['help']} | {r['harm']} | {r['oracle_lead']} | {r['null_lead']} | {r['null_lead_with_sign']} |" for r in rep["logs"]]
    out = ROOT / "experiments/results/null_lead"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

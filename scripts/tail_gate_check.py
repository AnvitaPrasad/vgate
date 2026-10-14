"""PREREG v10: does a non-linear benefit model find the clean block that a linear one does not? (EXPLORATORY)

Gradient-boosted trees against logistic regression, each fitted out of fold on one benchmark of the main reader's held-out
log (demo/sample_log.csv). Rule: read iff P(help) - P(harm) > 0. Reported per benchmark: help-vs-harm AUROC, the rule's lead
over the better fixed policy with a 95% image-cluster interval, the best lead any threshold reaches (chosen with the labels),
and the ceiling and the estimate for that AUROC (src/vgate/gate.py).

  python scripts/tail_gate_check.py   -> experiments/results/tail_gate_check.{json,md}
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vgate import gate  # noqa: E402

MODELS = dict(logistic=lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)),
              boosted=lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                                             early_stopping=False, random_state=0))


def oof(make, X, u, groups):
    out = np.zeros(len(u))
    for tr, te in GroupKFold(n_splits=5).split(X, u, groups):
        m = make().fit(X[tr], u[tr])
        p, cls = m.predict_proba(X[te]), list(m.classes_)
        out[te] = (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)
    return out


def main():
    d = pd.read_csv(ROOT / "demo/sample_log.csv")
    feats = [c for c in d.select_dtypes("number") if c not in ("c_ne", "c_ev", "group", "segment", "cost_ne", "cost_ev", "decision")]
    rep, L = dict(n_features=len(feats), benchmarks={}), []
    for b, k in d.groupby("segment"):
        X, u, g = np.nan_to_num(k[feats].to_numpy(float)), (k.c_ev - k.c_ne).to_numpy().astype(int), k["group"].to_numpy()
        h, m = float((u == 1).mean()), float((u == -1).mean())
        fixed = max(0.0, float(u.mean()))
        cell = dict(n=len(k), help=round(100 * h, 1), harm=round(100 * m, 1), auroc_floor=round(gate.auroc_needed(h, m, shape="any"), 3),
                    auroc_needed=round(gate.auroc_needed(h, m), 3), models={})
        for name, make in MODELS.items():
            s = oof(make, X, u, g)
            a = gate.auroc(s[u != 0], u[u != 0] > 0)
            read = s > 0
            pt, lo, hi = gate.cluster_bootstrap(g, lambda i: float((u[i] * read[i]).mean() - max(0.0, u[i].mean())), n_boot=2000, alpha=0.05)
            o = np.argsort(-s, kind="stable")
            best = float((np.concatenate([[0.0], np.cumsum(u[o])]) / len(u)).max() - fixed)
            cell["models"][name] = dict(auroc=round(a, 3), read_rate=round(100 * float(read.mean()), 1), rule_lead=round(100 * pt, 2), lo=round(100 * lo, 2),
                                        hi=round(100 * hi, 2), best_threshold_lead=round(100 * best, 2), estimate=round(100 * gate.gain_estimate(h, m, a), 2),
                                        ceiling=round(100 * gate.gain_ceiling(h, m, a), 2))
        cell["T1_boosted_rule_lead_below_one_point"] = bool(cell["models"]["boosted"]["rule_lead"] < 1.0)
        cell["T2_auroc_within_0p03"] = bool(abs(cell["models"]["boosted"]["auroc"] - cell["models"]["logistic"]["auroc"]) <= 0.03)
        cell["go"] = bool(cell["models"]["boosted"]["rule_lead"] >= 1.0 and cell["models"]["boosted"]["lo"] > 0)
        rep["benchmarks"][b] = cell
    rep["T1"] = all(c["T1_boosted_rule_lead_below_one_point"] for c in rep["benchmarks"].values())
    rep["T2"] = all(c["T2_auroc_within_0p03"] for c in rep["benchmarks"].values())
    rep["go"] = any(c["go"] for c in rep["benchmarks"].values())
    L += ["# PREREG v10 — gradient-boosted trees against logistic regression on the main reader's held-out log (exploratory)", "",
          f"{rep['n_features']} signals; five folds out of fold, image groups kept together; rule: read iff P(help) - P(harm) > 0; 95% image-cluster intervals.", "",
          "| benchmark | helps % | harms % | AUROC floor / needed | model | AUROC | read % | rule's lead [95%] | best threshold | estimate | ceiling |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, c in rep["benchmarks"].items():
        for name, v in c["models"].items():
            L.append(f"| {b} | {c['help']} | {c['harm']} | {c['auroc_floor']} / {c['auroc_needed']} | {name} | {v['auroc']} | {v['read_rate']} | "
                     f"{v['rule_lead']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] | {v['best_threshold_lead']:+.2f} | {v['estimate']:+.2f} | {v['ceiling']:+.2f} |")
    L += ["", f"T1 (boosted rule's lead below one point on every benchmark): {'held' if rep['T1'] else 'NOT held'}. "
          f"T2 (AUROC within 0.03 of the logistic model's on every benchmark): {'held' if rep['T2'] else 'NOT held'}. "
          f"Replication on the other readers triggered: {'yes' if rep['go'] else 'no'}."]
    out = ROOT / "experiments/results/tail_gate_check"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1)); out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

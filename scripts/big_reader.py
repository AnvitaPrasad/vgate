"""PREREG v13: a larger reader (Qwen3-VL-32B-Instruct, 4-bit) on a frozen sample of the held-out questions.

Needs experiments/runs/J23 scored per benchmark:
  python scripts/score.py --run experiments/runs/J23 --dataset <evqa|infoseek|okvqa> --model Qwen3-VL-32B-Instruct
Then:
  python scripts/big_reader.py   -> experiments/results/big_reader.{json,md} and the log experiments/results/big_reader__log.csv

Per benchmark: the two accuracies, the help and harm shares, the planner's floor, needed AUROC and null lead. On the three
benchmarks as mixed traffic: gate.audit with gates fitted out of fold on this log (uncertainty and retriever-score features
plus 64 components of the closed-book hidden state; scaler and PCA fitted on the log, without labels), 95% image-cluster
intervals. There is no development split for this reader: every gate is out of fold on held-out questions.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import operating_point as OP  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

M, RUN = "Qwen3-VL-32B-Instruct", ROOT / "experiments/runs/J23"
DATASETS = ("evqa", "infoseek", "okvqa")


def main(n_boot=2000):
    parts, H = [], []
    for d in DATASETS:
        s = A.load(RUN, ROOT / "data", d, M, "big")
        h = U.load_hidden(RUN, d, M)
        H.append(h[sorted(h)[len(h) // 2]].reindex(s.index).to_numpy())
        parts.append(pd.DataFrame(dict(segment=d, group=s.group, c_ne=s.c_ne.astype(int), c_ev=s.c_ev.astype(int), cost_ne=s.tok_ne, cost_ev=s.tok_ev,
                                       **{f: s[f] for f in OP.BASE})))
    H = np.vstack(H); assert np.isfinite(H).all(), "non-finite hidden states: repair the run first (see PREREG, note on v7d)"
    log = pd.concat(parts, ignore_index=True)
    k = min(64, len(log) - 1)
    log[[f"h{j}" for j in range(k)]] = PCA(n_components=k, random_state=0).fit_transform(StandardScaler().fit_transform(H)).round(4)
    log.to_csv(ROOT / "experiments/results/big_reader__log.csv", index=False, lineterminator="\n")
    feats = list(OP.BASE) + [f"h{j}" for j in range(k)]
    X, u, g, seg = np.nan_to_num(log[feats].to_numpy(float)), (log.c_ev - log.c_ne).to_numpy(), log.group.to_numpy(), log.segment.to_numpy()
    rep = dict(model=M, quantisation="4-bit NF4, half-precision compute", n=len(log), benchmarks={})
    for d in DATASETS:
        kk = seg == d; ud = u[kk]
        h, m = float((ud == 1).mean()), float((ud == -1).mean())
        s = gate.oof_scores(X[kk], ud, g[kk], kind="benefit")  # a benefit gate fitted out of fold on this benchmark alone
        read = s > 0
        pt, lo, hi = gate.cluster_bootstrap(g[kk], lambda i: float((ud[i] * read[i]).mean() - max(0.0, ud[i].mean())), n_boot=n_boot, alpha=0.05)
        rep["benchmarks"][d] = dict(n=int(kk.sum()), never=round(100 * float(log.c_ne[kk].mean()), 1), always=round(100 * float(log.c_ev[kk].mean()), 1),
                                    help=round(100 * h, 1), harm=round(100 * m, 1), auroc_floor=round(gate.auroc_needed(h, m, shape="any"), 3),
                                    auroc_needed=round(gate.auroc_needed(h, m), 3), null_lead=round(100 * gate.null_lead(h, m, int(kk.sum())), 2),
                                    gate_auroc=round(gate.auroc(s[ud != 0], ud[ud != 0] > 0), 3),
                                    gate_lead=[round(100 * x, 2) for x in (pt, lo, hi)])
    aud = gate.audit(X, log.c_ne, log.c_ev, g, seg, n_boot=n_boot)
    rep["audit"] = {r.policy: dict(accuracy=float(r.accuracy), read=float(r.read), vs_told=[float(r.vs_fixed), float(r.lo), float(r.hi)]) for r in aud.itertuples()}
    B, au = rep["benchmarks"], rep["audit"]
    told, ben, cls = au["fixed per segment (needs the label)"], au["benefit gate"], au["segment predicted"]
    rep["Q1"] = bool(B["evqa"]["always"] > B["evqa"]["never"] and B["infoseek"]["always"] > B["infoseek"]["never"] and B["okvqa"]["always"] < B["okvqa"]["never"])
    rep["Q2"] = bool(ben["vs_told"][0] <= 0)
    rep["Q3"] = bool(abs(cls["accuracy"] - ben["accuracy"]) <= 1.0)
    rep["Q4"] = bool(not any(v["gate_lead"][0] >= 1.0 and v["gate_lead"][1] > 0 for v in B.values()))
    L = [f"# PREREG v13 — a larger reader: {M}, 4-bit, on {len(log)} held-out questions ({', '.join(str(v['n']) + ' ' + d for d, v in B.items())})", "",
         "| benchmark | never | always | helps % | harms % | AUROC floor | needed | null lead | gate (out of fold): AUROC | its lead over the better fixed policy [95%] |", "|---|---|---|---|---|---|---|---|---|---|"]
    L += [f"| {d} | {v['never']} | {v['always']} | {v['help']} | {v['harm']} | {v['auroc_floor']} | {v['auroc_needed']} | {v['null_lead']} | {v['gate_auroc']} | "
          f"{v['gate_lead'][0]:+.2f} [{v['gate_lead'][1]:+.2f}, {v['gate_lead'][2]:+.2f}] |" for d, v in B.items()]
    L += ["", "Mixed traffic (accuracy = mean over the three benchmarks; gates fitted out of fold on this log):", "",
          "| policy | accuracy | read % | vs told the benchmark [95%] |", "|---|---|---|---|"]
    L += [f"| {p} | {v['accuracy']} | {v['read']} | {v['vs_told'][0]:+.1f} [{v['vs_told'][1]:+.1f}, {v['vs_told'][2]:+.1f}] |" for p, v in au.items()]
    L += ["", "Expectations: " + "; ".join(f"{q} {'held' if rep[q] else 'NOT held'}" for q in ("Q1", "Q2", "Q3", "Q4")) + "."]
    out = ROOT / "experiments/results/big_reader"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

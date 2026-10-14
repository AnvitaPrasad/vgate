"""PREREG v10b (EXPLORATORY, not blind): a boosted gate fitted on development data, in the paper's own protocol.

Per reader and benchmark: VGate-full's features (operating_point.fit_benefit_gates: free and retriever-score features plus 64
components of the closed-book hidden state, scaler and PCA fitted on that benchmark's development questions), with the paper's
logistic model and with the boosted trees of PREREG v10 in its place; fitted on the development questions, applied once to
the held-out questions; rule: read iff P(help) - P(harm) > 0. Reported: the lead over the better fixed policy (95%
image-cluster interval), the help-vs-harm AUROC, the planner's estimate, and the lead on the held-out questions whose question
string occurs in the development set and on those whose string does not.

  python scripts/tail_gate_dev.py   -> experiments/results/tail_gate_dev.{json,md}
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import lodo  # noqa: E402
import operating_point as OP  # noqa: E402
import tail_gate_check as T  # noqa: E402
from vgate import gate  # noqa: E402

norm = lambda q: " ".join(q.lower().split())


def strings(path, qids):
    recs = {r["qid"]: r for r in map(json.loads, open(path))}
    return np.array([norm(recs[q]["question"]) for q in qids])


def cell(tr, Htr, te, Hte, s_tr, s_te):
    G = OP.fit_benefit_gates(tr, Htr)
    P = lambda H: G["pca"].transform(G["sc"].transform(H))
    Xtr = np.hstack([np.nan_to_num(tr[OP.BASE].to_numpy()), P(Htr)]); Xte = np.hstack([np.nan_to_num(te[OP.BASE].to_numpy()), P(Hte)])
    u_tr, u, g = tr.u.to_numpy().astype(int), te.u.to_numpy().astype(int), te.group.to_numpy()
    h, m = float((u == 1).mean()), float((u == -1).mean())
    always = h > m  # the held-out split's better fixed policy
    seen = np.isin(s_te, np.unique(s_tr))
    bst = T.MODELS["boosted"]().fit(Xtr, u_tr)
    p, cls = bst.predict_proba(Xte), list(bst.classes_)
    S = dict(logistic=OP.benefit_score(G["full"], Xte), boosted=(p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0))
    out = dict(n_dev=len(tr), n=len(te), help=round(100 * h, 1), harm=round(100 * m, 1), better_fixed="always" if always else "never",
               auroc_needed=round(gate.auroc_needed(h, m), 3), share_seen_strings=round(100 * float(seen.mean()), 1), models={})
    for name, s in S.items():
        d = u * (s > 0) - (u if always else 0)  # per question: the gate's answer minus the fixed policy's

        def lead(k):
            if k.sum() < 20:
                return None
            pt, lo, hi = gate.cluster_bootstrap(g[k], lambda i: float(d[k][i].mean()), n_boot=2000, alpha=0.05)
            return dict(n=int(k.sum()), lead=round(100 * pt, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2))
        a = gate.auroc(s[u != 0], u[u != 0] > 0)
        out["models"][name] = dict(lead(np.ones(len(u), bool)), auroc=round(a, 3), estimate=round(100 * gate.gain_estimate(h, m, a), 2),
                                   read_rate=round(100 * float((s > 0).mean()), 1), seen=lead(seen), unseen=lead(~seen))
    return out


def main():
    rep = {}
    for model in lodo.MODELS:
        Fd, Hd = lodo.load(model, "dev"); Ft, Ht = lodo.load(model, "confirm")
        rep[model] = {d: cell(Fd[d], Hd[d], Ft[d], Ht[d], strings(ROOT / "data/processed" / f"{d}.jsonl", Fd[d].index),
                              strings(ROOT / "data/processed" / f"{d}.jsonl", Ft[d].index)) for d in Fd}
    import cvqa_audit as CA
    data = ROOT / "experiments/runs/J14/data"
    if (ROOT / "experiments/runs/J15" / f"scores__cvqa__{CA.MODEL}.csv").exists():
        tr, Htr, _ = CA.load(ROOT / "experiments/runs/J14", data, "dev"); te, Hte, _ = CA.load(ROOT / "experiments/runs/J15", data, "confirm")
        rep[CA.MODEL]["cvqa"] = cell(tr, Htr, te, Hte, strings(data / "processed/cvqa.jsonl", tr.index), strings(data / "processed/cvqa.jsonl", te.index))
    f = lambda v: "n/a" if v is None else f"{v['lead']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]"
    L = ["# PREREG v10b — gates fitted on development data and applied to held-out data: logistic (the paper's VGate-full) and boosted trees (exploratory, not blind)", "",
         "Lead of the rule \"read iff P(help) - P(harm) > 0\" over the held-out split's better fixed policy, points, 95% image-cluster intervals. "
         "Seen / unseen: held-out questions whose question string does / does not occur in the development set (lead within that subset).", "",
         "| reader | benchmark | model | AUROC | needed | estimate | lead | seen strings: share, lead | unseen strings: lead |", "|---|---|---|---|---|---|---|---|---|"]
    for model, R_ in rep.items():
        for d, c in R_.items():
            for name, v in c["models"].items():
                L.append(f"| {model} | {d} | {name} | {v['auroc']} | {c['auroc_needed']} | {v['estimate']:+.2f} | {f(v)} | {c['share_seen_strings']}%, {f(v['seen'])} | {f(v['unseen'])} |")
    wins = [(m_, d, n_) for m_, R_ in rep.items() for d, c in R_.items() for n_, v in c["models"].items() if v["lead"] >= 1.0 and v["lo"] > 0]
    L += ["", f"Leads of at least one point with an interval above zero: {wins or 'none'}."]
    out = ROOT / "experiments/results/tail_gate_dev"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

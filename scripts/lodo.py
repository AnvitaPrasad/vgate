"""Gates on a benchmark they were not fitted on (PREREG v4a): fit on the dev sets of two benchmarks, apply to the third.

  python scripts/lodo.py --split confirm   -> experiments/results/lodo__confirm.{json,md}
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import operating_point as OP  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

R, DATA = ROOT / "experiments/runs", ROOT / "data"
MODELS = {"Qwen3-VL-4B-Instruct": dict(dev=dict(evqa="J1", infoseek="J1", okvqa="J6"), confirm=dict(evqa="J4", infoseek="J4", okvqa="J6")),
          "Qwen2.5-VL-3B-Instruct": {s: dict(evqa="J2", infoseek="J2", okvqa="J6") for s in ("dev", "confirm")},
          # PREREG v4c (models 3 and 4; OK-VQA from J11 / J12)
          "InternVL3_5-4B-HF": {s: dict(evqa="J5", infoseek="J5", okvqa="J11") for s in ("dev", "confirm")},
          "Qwen3-VL-8B-Instruct": {s: dict(evqa="J7", infoseek="J7", okvqa="J12") for s in ("dev", "confirm")}}
V4A = ["Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct"]
RULES = ["never", "always", "train_majority", "nearest_regime", "error_post", "error_full", "benefit_post", "benefit_pre", "benefit_full", "oracle"]
GATES = ["nearest_regime", "error_post", "error_full", "benefit_post", "benefit_pre", "benefit_full"]


def load(model, split):
    fr, H = {}, {}
    for d, run in MODELS[model][split].items():
        s = A.load(R / run, DATA, d, model, split)
        h = U.load_hidden(R / run, d, model)
        fr[d], H[d] = s, h[sorted(h)[len(h) // 2]].reindex(s.index).to_numpy()
    return fr, H


def decide(tr, Htr, te, Hte):
    """Read decisions on `te` of every rule, from models fit on `tr` (two benchmarks, no dataset identity in the gates)."""
    lr = lambda c=1.0: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    G = OP.fit_benefit_gates(tr, Htr)
    Ptr, Pte = G["pca"].transform(G["sc"].transform(Htr)), G["pca"].transform(G["sc"].transform(Hte))
    Xtr, Xte = np.nan_to_num(tr[OP.BASE].to_numpy()), np.nan_to_num(te[OP.BASE].to_numpy())
    wrong, u, ds = (1 - tr.c_ne).to_numpy(), tr.u.to_numpy(), tr.group.str.split("/").str[0].to_numpy()

    def p_wrong(Ftr, Fte):
        m = lr().fit(Ftr, wrong)
        return m.predict_proba(Fte)[:, list(m[-1].classes_).index(1)]
    better = {o: bool(tr.c_ev[ds == o].mean() > tr.c_ne[ds == o].mean()) for o in np.unique(ds)}
    majority = np.mean([tr.c_ev[ds == o].mean() - tr.c_ne[ds == o].mean() for o in better]) > 0
    nearest = lr(0.1).fit(Ptr, ds).predict(Pte)
    n = len(te)
    return dict(never=np.zeros(n, bool), always=np.ones(n, bool), train_majority=np.full(n, majority),
                nearest_regime=np.array([better[o] for o in nearest]),
                error_post=p_wrong(Xtr, Xte) > 0.5, error_full=p_wrong(np.hstack([Xtr, Ptr]), np.hstack([Xte, Pte])) > 0.5,
                benefit_post=OP.benefit_score(lr().fit(Xtr, u), Xte) > 0, benefit_pre=OP.benefit_score(G["pre"], Pte) > 0,
                benefit_full=OP.benefit_score(G["full"], np.hstack([Xte, Pte])) > 0, oracle=te.u.to_numpy() > 0)


def run(model, split, n_boot):
    tr_fr, tr_H = load(model, "dev")
    te_fr, te_H = (tr_fr, tr_H) if split == "dev" else load(model, split)
    names = list(tr_fr)
    acc, rep = {}, {}
    for d in names:
        others = [o for o in names if o != d]
        D = decide(pd.concat([tr_fr[o] for o in others]), np.vstack([tr_H[o] for o in others]), te_fr[d], te_H[d])
        te = te_fr[d]
        acc[d] = {k: np.where(D[k], te.c_ev.to_numpy(), te.c_ne.to_numpy()).astype(float) for k in RULES}
        acc[d]["best_fixed"] = acc[d]["always"] if acc[d]["always"].mean() >= acc[d]["never"].mean() else acc[d]["never"]
        rep[d] = {k: dict(acc=round(100 * float(acc[d][k].mean()), 2), read=round(100 * float(D[k].mean()), 1)) for k in RULES}
        rep[d]["best_fixed"] = dict(acc=round(100 * float(acc[d]["best_fixed"].mean()), 2))
    rep["macro"] = {k: round(float(np.mean([rep[d][k]["acc"] for d in names])), 2) for k in RULES + ["best_fixed"]}
    groups = np.concatenate([te_fr[d].group.to_numpy() for d in names])
    strata = np.concatenate([np.full(len(te_fr[d]), j) for j, d in enumerate(names)])
    offs = np.cumsum([0] + [len(te_fr[d]) for d in names])

    def diff(a, b):
        def stat(idx):
            return float(np.mean([(acc[d][a][i] - acc[d][b][i]).mean() for d, lo, hi in zip(names, offs[:-1], offs[1:])
                                  for i in [idx[(idx >= lo) & (idx < hi)] - lo]]))
        return [round(100 * v, 2) for v in gate.cluster_bootstrap(groups, stat, n_boot=n_boot, alpha=0.05, strata=strata)]
    rep["macro_diff_95ci"] = {"error_full - benefit_full": diff("error_full", "benefit_full"), "error_post - benefit_post": diff("error_post", "benefit_post"),
                              **{f"{g} - best_fixed": diff(g, "best_fixed") for g in GATES}, **{f"{g} - always": diff(g, "always") for g in GATES}}
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="confirm")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--models", default=",".join(V4A), help="comma-separated; default = the two models of PREREG v4a")
    ap.add_argument("--tag", default="", help="suffix of the output files, e.g. v4c")
    a = ap.parse_args()
    res = {m: run(m, a.split, a.n_boot) for m in a.models.split(",")}
    exp = dict(L1=all(res[m]["macro_diff_95ci"][f"{g} - best_fixed"][0] < -3.0 for m in res for g in GATES),
               L2=all(res[m]["macro_diff_95ci"]["error_full - benefit_full"][1] > -0.5 for m in res),
               L3=all(res[m]["okvqa"][g]["read"] > 50 for m in res for g in GATES))
    lines = [f"# Gates on a benchmark they were not fitted on ({a.split} split; PREREG v4a)", "",
             "Accuracy % (share of queries read %) on the benchmark named in the column; gates fitted on the dev sets of the other two.", ""]
    for m, r in res.items():
        ds = [d for d in r if d not in ("macro", "macro_diff_95ci")]
        lines += [f"## {m}", "", "| rule | " + " | ".join(ds) + " | macro |", "|---|" + "---|" * (len(ds) + 1)]
        for k in RULES[:-1] + ["best_fixed", "oracle"]:
            cell = lambda d: f"{r[d][k]['acc']:.1f}" + (f" ({r[d][k]['read']:.0f})" if "read" in r[d][k] else "")
            lines.append(f"| {k} | " + " | ".join(cell(d) for d in ds) + f" | {r['macro'][k]:.1f} |")
        f = lambda v: f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"
        lines += ["", "Macro differences (95%): " + "; ".join(f"{k} {f(v)}" for k, v in r["macro_diff_95ci"].items() if "always" not in k) + ".", ""]
    lines += ["## Expectations (PREREG v4a)", ""] + [f"- {k}: {'held' if v else 'NOT held'}" for k, v in exp.items()]
    out = ROOT / "experiments/results" / (f"lodo__{a.split}" + (f"__{a.tag}" if a.tag else ""))
    out.with_suffix(".json").write_text(json.dumps(dict(split=a.split, results=res, expectations=exp), indent=1))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

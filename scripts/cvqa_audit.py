"""CVQA as a fourth benchmark (PREREG v7): the paper's gates fitted on the development split, frozen, and applied to the
held-out split of a benchmark chosen because published numbers say retrieval both helps and hurts there. Written before
any CVQA output exists.

  python scripts/cvqa_audit.py --split dev       # regime check R1 and out-of-fold estimates (development data, 80% intervals)
  python scripts/cvqa_audit.py --split confirm   # the registered tests G1-G3 (95% intervals)
      -> experiments/results/cvqa_audit__<split>.{json,md}
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import operating_point as OP  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

MODEL, SEGS = "Qwen3-VL-4B-Instruct", ("category", "subset")
RULES = ["never", "always", "best_fixed", "told_category", "category_classifier", "told_subset", "subset_classifier",
         "error_post", "error_full", "benefit_post", "benefit_pre", "benefit_full", "oracle"]
NAMES = dict(never="never retrieve", always="always retrieve", best_fixed="better fixed policy (chosen on development data)",
             told_category="told the category (10)", category_classifier="category predicted from the gate's features",
             told_subset="told the country-language subset (39)", subset_classifier="subset predicted from the gate's features",
             error_post="error gate, free + retriever-score features", error_full="error gate, VGate-full's features",
             benefit_post="VGate-post", benefit_pre="VGate-pre", benefit_full="VGate-full", oracle="oracle gate")


def load(run, data, split, fallback=False):
    fr = A.load(run, data, "cvqa", MODEL, split)
    sc = pd.read_csv(pathlib.Path(run) / f"scores__cvqa__{MODEL}.csv", index_col="qid")
    sc.index = sc.index.astype(str)
    fr["no_letter_ev"] = sc.letter_ev.reindex(fr.index).fillna("").eq("").to_numpy()  # the evidence reply names no option
    if fallback:  # sensitivity analysis: such a reply falls back to the closed-book answer
        fr.loc[fr.no_letter_ev, "c_ev"] = fr.c_ne[fr.no_letter_ev]
        fr["u"] = (fr.c_ev - fr.c_ne).astype(int)
    h = U.load_hidden(run, "cvqa", MODEL)
    H = h[sorted(h)[len(h) // 2]].reindex(fr.index).to_numpy()
    recs = {r["qid"]: r for r in map(json.loads, open(pathlib.Path(data) / "processed" / "cvqa.jsonl"))}
    seg = pd.DataFrame(dict(category=[recs[q]["qtype"] for q in fr.index], subset=[recs[q]["subset"] for q in fr.index]), index=fr.index)
    return fr, H, seg


def decide(tr, Htr, te, Hte, seg_tr, seg_te):
    """Read decisions (and the two equal-feature scores) on `te` from models fit on `tr` only."""
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    G = OP.fit_benefit_gates(tr, Htr)
    Ptr, Pte = G["pca"].transform(G["sc"].transform(Htr)), G["pca"].transform(G["sc"].transform(Hte))
    Xtr, Xte = np.nan_to_num(tr[OP.BASE].to_numpy()), np.nan_to_num(te[OP.BASE].to_numpy())
    Ftr, Fte = np.hstack([Xtr, Ptr]), np.hstack([Xte, Pte])
    wrong, u, c0, c1 = (1 - tr.c_ne).to_numpy().astype(int), tr.u.to_numpy(), tr.c_ne.to_numpy(), tr.c_ev.to_numpy()

    def p_wrong(a, b):
        m = lr().fit(a, wrong)
        return m.predict_proba(b)[:, list(m[-1].classes_).index(1)]
    best, n = bool(c1.mean() > c0.mean()), len(te)
    S = dict(error_post=p_wrong(Xtr, Xte), error_full=p_wrong(Ftr, Fte), benefit_post=OP.benefit_score(lr().fit(Xtr, u), Xte),
             benefit_pre=OP.benefit_score(G["pre"], Pte), benefit_full=OP.benefit_score(G["full"], Fte))
    D = dict(never=np.zeros(n, bool), always=np.ones(n, bool), best_fixed=np.full(n, best), oracle=te.u.to_numpy() > 0)
    D.update({k: S[k] > (0.5 if k.startswith("error") else 0) for k in S})
    for name in SEGS:  # the better fixed policy of each segment, with the segment given or predicted from the gate's features
        s_tr = seg_tr[name].to_numpy()
        pol = {s: bool(c1[s_tr == s].mean() > c0[s_tr == s].mean()) for s in np.unique(s_tr)}
        D[f"told_{name}"] = np.array([pol.get(s, best) for s in seg_te[name]])
        D[f"{name}_classifier"] = np.array([pol.get(s, best) for s in lr().fit(Ftr, s_tr).predict(Fte)])
    return D, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "confirm"])
    ap.add_argument("--run_dev", default=str(ROOT / "experiments/runs/J14"))
    ap.add_argument("--run_confirm", default=str(ROOT / "experiments/runs/J15"))
    ap.add_argument("--data", default=str(ROOT / "experiments/runs/J14/data"))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "experiments/results"))
    ap.add_argument("--fallback", action="store_true", help="an evidence reply that names no option falls back to the closed-book answer")
    ap.add_argument("--model", default=MODEL, help="reader whose outputs are analysed (PREREG v7d: a second reader)")
    a = ap.parse_args()
    tag = "" if a.model == MODEL else f"__{a.model}"
    globals()["MODEL"] = a.model
    tr, Htr, seg_tr = load(a.run_dev, a.data, "dev", a.fallback)
    if a.split == "dev":  # out of fold over the development split, image groups kept together
        te, seg_te, alpha = tr, seg_tr, 0.2
        D = {k: np.zeros(len(te), bool) for k in RULES}
        S = {k: np.zeros(len(te)) for k in ("error_post", "error_full", "benefit_post", "benefit_pre", "benefit_full")}
        for i, j in GroupKFold(n_splits=5).split(Htr, groups=tr.group.to_numpy()):
            d, s = decide(tr.iloc[i], Htr[i], tr.iloc[j], Htr[j], seg_tr.iloc[i], seg_tr.iloc[j])
            for k in D:
                D[k][j] = d[k]
            for k in S:
                S[k][j] = s[k]
    else:
        te, Hte, seg_te = load(a.run_confirm, a.data, "confirm", a.fallback)
        assert not set(te.group) & set(tr.group), "an image is in both splits"
        alpha = 0.05
        D, S = decide(tr, Htr, te, Hte, seg_tr, seg_te)
    c0, c1, grp = te.c_ne.to_numpy().astype(float), te.c_ev.to_numpy().astype(float), te.group.to_numpy()
    acc = {k: np.where(D[k], c1, c0) for k in RULES}
    zero = np.zeros(len(te))

    def diff(x, y):
        return [round(100 * v, 2) for v in gate.cluster_bootstrap(grp, lambda i: float(acc[x][i].mean() - acc[y][i].mean()), n_boot=a.n_boot, alpha=alpha)]

    def nauc_diff(x, y):
        f = lambda k, i: gate.nauc(gate.curve(S[k][i], c0[i], c1[i], zero[i], zero[i])[0])
        return [round(100 * v, 2) for v in gate.cluster_bootstrap(grp, lambda i: f(x, i) - f(y, i), n_boot=a.n_boot, alpha=alpha)]
    rep = dict(split=a.split, n=len(te), n_fit=len(tr), interval=f"{round(100 * (1 - alpha))}%", fallback=bool(a.fallback),
               no_letter=round(100 * float(te.no_letter_ev.mean()), 2), no_letter_among_harm=round(100 * float(te.no_letter_ev[(te.c_ne == 1) & (te.c_ev == 0)].mean()), 2) if not a.fallback else None,
               help=round(100 * float((te.u == 1).mean()), 2), harm=round(100 * float((te.u == -1).mean()), 2),
               rules={k: dict(acc=round(100 * float(acc[k].mean()), 2), read=round(100 * float(D[k].mean()), 1)) for k in RULES},
               oracle_headroom=round(100 * float(acc["oracle"].mean() - max(c0.mean(), c1.mean())), 2),
               full_minus_best_fixed=diff("benefit_full", "best_fixed"), full_minus_never=diff("benefit_full", "never"),
               full_minus_always=diff("benefit_full", "always"), full_minus_error_full=diff("benefit_full", "error_full"),
               post_minus_error_post=diff("benefit_post", "error_post"), nauc_full_minus_error_full=nauc_diff("benefit_full", "error_full"),
               full_minus_told_category=diff("benefit_full", "told_category"), full_minus_told_subset=diff("benefit_full", "told_subset"),
               full_minus_category_classifier=diff("benefit_full", "category_classifier"))
    if a.split == "dev":
        rep["expectations"] = dict(R1=bool(rep["help"] >= 8 and rep["harm"] >= 8))
    else:
        g = lambda k: rep[k]
        rep["expectations"] = dict(G1=bool(g("full_minus_best_fixed")[0] >= 1.0 and g("full_minus_best_fixed")[1] > 0),
                                   G2=bool(g("full_minus_error_full")[1] > 0), G3=bool(g("full_minus_told_category")[1] > 0))
    f = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
    L = [f"# CVQA, a fourth benchmark (PREREG v7) — {a.split} split, n = {rep['n']} (gates fitted on {rep['n_fit']} development questions"
         + (", out of fold" if a.split == "dev" else "") + f"; {rep['interval']} image-cluster intervals)", "",
         f"Retrieval helps on {rep['help']}% of questions and hurts on {rep['harm']}%; an oracle gate is {rep['oracle_headroom']} points above the better fixed policy."
         + (" Labels: an evidence reply that names no option falls back to the closed-book answer." if a.fallback else
            f" The evidence reply names no option on {rep['no_letter']}% of questions ({rep['no_letter_among_harm']}% of the harm cases)."), "",
         "| Policy | Accuracy | Read (%) |", "|---|---|---|"] + [f"| {NAMES[k]} | {rep['rules'][k]['acc']} | {rep['rules'][k]['read']} |" for k in RULES] + [""]
    for k, name in (("full_minus_best_fixed", "VGate-full − better fixed policy"), ("full_minus_never", "VGate-full − never"), ("full_minus_always", "VGate-full − always"),
                    ("full_minus_error_full", "VGate-full − error gate with the same features"), ("post_minus_error_post", "VGate-post − error gate with the same features"),
                    ("nauc_full_minus_error_full", "nAUC, benefit − error label (VGate-full's features)"), ("full_minus_told_category", "VGate-full − told the category"),
                    ("full_minus_told_subset", "VGate-full − told the subset"), ("full_minus_category_classifier", "VGate-full − category classifier")):
        L.append(f"- {name}: {f(rep[k])}")
    L += ["", "## Expectations (PREREG v7)", ""] + [f"- {k}: {'held' if v else 'NOT held'}" for k, v in rep["expectations"].items()]
    out = pathlib.Path(a.out) / (f"cvqa_audit__{a.split}" + ("__fallback" if a.fallback else "") + tag)
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

"""PREREG v2e (exploratory): budget-free operating points and mixed traffic.

  natural-threshold rules   benefit rule: read iff P(u=+1) - P(u=-1) > 0   (VGate-post, VGate-full, VGate-pre)
                            error rule:   read iff P(a_0 wrong) > 0.5      (error gate with VGate-post's features)
  per-dataset gates         fit on each dataset's dev split, as in confirm.py / extend.py
  mixed-traffic gate        ONE gate fit on the union of the dev splits, no dataset-identity feature, applied to the
                            union of the held-out splits

  python scripts/operating_point.py --model Qwen3-VL-4B-Instruct --runs evqa=J1:J4,infoseek=J1:J4,okvqa=J6:J6
Writes experiments/results/operating_point__<model>[__<labels>].{json,md}. Nothing is tuned on held-out data.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import confirm as C  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

BASE = C.FEATS["post"]
RULES = ["never", "always", "fixed_by_dataset", "router", "error_rule", "error_rule_post_only", "error_rule_full",
         "benefit_rule_post", "benefit_rule_full", "benefit_rule_pre", "cascade", "error_tuned", "error_full_tuned",
         "benefit_full_tuned", "oracle"]
PRE_SEARCH = {"never", "fixed_by_dataset", "router", "benefit_rule_pre", "cascade"}  # these decide before searching
# Reference policies added after an independent review (2026-10-04), to separate regime detection from per-query gating:
#   fixed_by_dataset  told the dataset: the better of never / always on that dataset's dev split
#   router            predicts the dataset from the closed-book hidden state, then applies that dataset's fixed policy
#   cascade           the system as drawn: search iff VGate-pre's benefit > 0, then read iff VGate-full's benefit > 0
# error_rule_full / error_full_tuned: the error target with VGate-full's features (equal-feature counterpart).
# *_tuned (robustness check added after the v2e results were seen): the same two gates with the threshold that
# maximises out-of-fold dev accuracy (macro over datasets) instead of the natural thresholds 0.5 / 0.


def fit_benefit_gates(tr, Htr):
    """The two benefit gates of the system, fit on `tr`: VGate-pre (PCA-64 of the hidden state, C=0.1) and VGate-full
    ([free + QPP features, PCA-64 of the hidden state], C=1). Returns the fitted pieces; `decisions` and
    scripts/freeze_gates.py use exactly these objects."""
    sc = StandardScaler().fit(Htr)
    pca = PCA(n_components=64, random_state=0).fit(sc.transform(Htr))
    Ptr, u = pca.transform(sc.transform(Htr)), tr.u.to_numpy()
    Xtr = np.nan_to_num(tr[BASE].to_numpy())
    pre = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Ptr, u)
    full = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(np.hstack([Xtr, Ptr]), u)
    return dict(sc=sc, pca=pca, pre=pre, full=full)


def benefit_score(m, F):
    p, cls = m.predict_proba(F), list(m[-1].classes_)
    return (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)


def decisions(tr, te, Htr, Hte):
    """Read decisions (bool arrays over `te`) of every rule, from models fit on `tr` only."""
    Xtr, Xte = np.nan_to_num(tr[BASE].to_numpy()), np.nan_to_num(te[BASE].to_numpy())
    u, wrong = tr.u.to_numpy(), (1 - tr.c_ne).to_numpy()
    G = fit_benefit_gates(tr, Htr)
    sc, pca = G["sc"], G["pca"]
    Ptr, Pte = pca.transform(sc.transform(Htr)), pca.transform(sc.transform(Hte))
    lr = lambda c=1.0: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))

    def benefit(Ftr, Fte, c=1.0):
        m = lr(c).fit(Ftr, u)
        p, cls = m.predict_proba(Fte), list(m[-1].classes_)
        return (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)

    m_err = lr().fit(Xtr, wrong)
    n = len(te)
    s_err = m_err.predict_proba(Xte)[:, list(m_err[-1].classes_).index(1)]
    Ftr, Fte = np.hstack([Xtr, Ptr]), np.hstack([Xte, Pte])
    s_full = benefit_score(G["full"], Fte)
    m_ef = lr().fit(Ftr, wrong)
    s_ef = m_ef.predict_proba(Fte)[:, list(m_ef[-1].classes_).index(1)]
    ds = tr.group.str.split("/").str[0].to_numpy()  # dataset of each dev row, for the macro average

    def tuned(F, y, kind, s_te):
        oof = gate.oof_scores(F, y, tr.group.to_numpy(), kind=kind)
        cand = np.unique(np.quantile(oof, np.linspace(0, 1, 201)))
        mac = [np.mean([np.where(oof[ds == d] >= t, tr.c_ev[ds == d], tr.c_ne[ds == d]).mean() for d in np.unique(ds)]) for t in cand]
        return s_te >= cand[int(np.argmax(mac))]
    ds_te = te.group.str.split("/").str[0].to_numpy()
    better = {d: bool(tr.c_ev[ds == d].mean() > tr.c_ne[ds == d].mean()) for d in np.unique(ds)}  # read always?
    if len(better) > 1:
        m_ds = lr(0.1).fit(Ptr, ds)
        pred_ds = m_ds.predict(Pte)
    else:
        pred_ds = ds_te
    pre = benefit_score(G["pre"], Pte) > 0  # hidden state only (C as in confirm.probe)
    return dict(never=np.zeros(n, bool), always=np.ones(n, bool),
                fixed_by_dataset=np.array([better[d] for d in ds_te]), router=np.array([better[d] for d in pred_ds]),
                router_acc=float((pred_ds == ds_te).mean()), cascade=pre & (s_full > 0),
                error_rule=s_err > 0.5, error_rule_post_only=s_err > 0.5, error_rule_full=s_ef > 0.5, error_full_tuned=tuned(Ftr, wrong, "binary", s_ef),
                error_tuned=tuned(Xtr, wrong, "binary", s_err), benefit_full_tuned=tuned(Ftr, u, "benefit", s_full),
                benefit_rule_post=benefit(Xtr, Xte) > 0,
                benefit_rule_full=s_full > 0,
                benefit_rule_pre=pre,
                oracle=te.u.to_numpy() > 0)


def evaluate(parts, n_boot):
    """parts: [(dataset, te, D)] -> per-dataset accuracy / read rate, macro average, paired CIs on the macro."""
    acc = lambda te, r: np.where(r, te.c_ev.to_numpy(), te.c_ne.to_numpy()).astype(float)
    srch = lambda D, k: float(D["benefit_rule_pre"].mean()) if k == "cascade" else float(D[k].mean()) if k in PRE_SEARCH else 1.0
    rep = {d: {k: dict(acc=float(acc(te, D[k]).mean()), read_rate=float(D[k].mean()), search_rate=srch(D, k)) for k in RULES}
           for d, te, D in parts}
    rep["macro"] = {k: dict(acc=float(np.mean([rep[d][k]["acc"] for d, _, _ in parts])),
                            read_rate=float(np.mean([rep[d][k]["read_rate"] for d, _, _ in parts])),
                            search_rate=float(np.mean([rep[d][k]["search_rate"] for d, _, _ in parts]))) for k in RULES}
    groups = np.concatenate([te.group.to_numpy() for _, te, _ in parts])
    strata = np.concatenate([np.full(len(te), j) for j, (_, te, _) in enumerate(parts)])
    offs = np.cumsum([0] + [len(te) for _, te, _ in parts])
    A_ = {k: [acc(te, D[k]) for _, te, D in parts] for k in RULES}

    def diff(a, b):
        def stat(idx):
            return float(np.mean([(A_[a][j][i] - A_[b][j][i]).mean() for j, (lo, hi) in enumerate(zip(offs[:-1], offs[1:]))
                                  for i in [idx[(idx >= lo) & (idx < hi)] - lo]]))
        return gate.cluster_bootstrap(groups, stat, n_boot=n_boot, alpha=0.05, strata=strata)
    rep["macro_diff_95ci"] = {f"{a} - {b}": diff(a, b) for a in ("benefit_rule_post", "benefit_rule_full", "benefit_rule_pre")
                              for b in ("error_rule", "always", "never")}
    rep["macro_diff_95ci"]["benefit_full_tuned - error_tuned"] = diff("benefit_full_tuned", "error_tuned")
    rep["macro_diff_95ci"]["benefit_rule_full - error_tuned"] = diff("benefit_rule_full", "error_tuned")
    rep["macro_diff_95ci"]["benefit_rule_full - error_rule_full"] = diff("benefit_rule_full", "error_rule_full")
    for a_ in ("benefit_rule_full", "benefit_rule_pre", "cascade"):
        for b_ in ("router", "fixed_by_dataset"):
            rep["macro_diff_95ci"][f"{a_} - {b_}"] = diff(a_, b_)
    rep["macro_diff_95ci"]["cascade - always"] = diff("cascade", "always")
    rep["macro_diff_95ci"]["cascade - error_rule_full"] = diff("cascade", "error_rule_full")
    rep["dataset_diff_95ci"] = {}
    for j, (d, te, D) in enumerate(parts):  # where does the margin over the error gate come from?
        x = A_["benefit_rule_full"][j] - A_["error_rule_full"][j]
        rep["dataset_diff_95ci"][d] = gate.cluster_bootstrap(te.group.to_numpy(), lambda i, x=x: float(x[i].mean()), n_boot=n_boot, alpha=0.05)
    rep["macro_diff_95ci"]["benefit_full_tuned - error_full_tuned"] = diff("benefit_full_tuned", "error_full_tuned")
    return rep


def table(rep, datasets, title):
    lines = [f"\n## {title}\n", "| rule | " + " | ".join(f"{d} acc (reads)" for d in datasets + ["macro"]) + " |",
             "|---|" + "---|" * (len(datasets) + 1)]
    for k in RULES:
        lines.append(f"| {k} | " + " | ".join(f"{100*rep[d][k]['acc']:.1f} ({100*rep[d][k]['read_rate']:.0f}%)" for d in datasets + ["macro"]) + " |")
    lines += ["\n| macro accuracy difference | point | 95% lo | 95% hi |", "|---|---|---|---|"]
    lines += [f"| {k} | {100*v[0]:+.2f} | {100*v[1]:+.2f} | {100*v[2]:+.2f} |" for k, v in rep["macro_diff_95ci"].items()]
    lines += ["\n| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |", "|---|---|---|---|"]
    lines += [f"| {d} | {100*v[0]:+.2f} | {100*v[1]:+.2f} | {100*v[2]:+.2f} |" for d, v in rep["dataset_diff_95ci"].items()]
    lines.append("\nSearch rates (macro): " + ", ".join(f"{k} {100*rep['macro'][k]['search_rate']:.0f}%" for k in ("benefit_rule_pre", "cascade", "router")))
    if "router_dataset_accuracy" in rep:
        lines.append(f"Router: dataset predicted from the closed-book hidden state with {100*rep['router_dataset_accuracy']:.1f}% accuracy.")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--runs", required=True, help="dataset=TRAINRUN:TESTRUN,... (folders under experiments/runs)")
    ap.add_argument("--labels", default="official", choices=["official", "relaxed"], help="relaxed: OK-VQA sensitivity label")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    spec = {x.split("=")[0]: x.split("=")[1].split(":") for x in a.runs.split(",")}
    datasets, R = list(spec), ROOT / "experiments/runs"
    tr, te, Htr, Hte = {}, {}, {}, {}
    for d, (r_tr, r_te) in spec.items():
        tr[d], te[d] = A.load(R / r_tr, ROOT / "data", d, a.model, "dev", a.labels), A.load(R / r_te, ROOT / "data", d, a.model, "confirm", a.labels)
        H1, H2 = U.load_hidden(R / r_tr, d, a.model), U.load_hidden(R / r_te, d, a.model)
        L = sorted(H1)[len(H1) // 2]
        Htr[d], Hte[d] = H1[L].reindex(tr[d].index).to_numpy(), H2[L].reindex(te[d].index).to_numpy()
    per = evaluate([(d, te[d], {k: v for k, v in decisions(tr[d], te[d], Htr[d], Hte[d]).items() if k != "router_acc"})
                    for d in datasets], a.n_boot)
    # mixed traffic: one gate on the pooled dev sets, applied to the pooled held-out sets, then split back per dataset
    import pandas as pd
    ptr, pte = pd.concat([tr[d] for d in datasets]), pd.concat([te[d] for d in datasets])
    Dm = decisions(ptr, pte, np.vstack([Htr[d] for d in datasets]), np.vstack([Hte[d] for d in datasets]))
    offs = np.cumsum([0] + [len(te[d]) for d in datasets])
    router_acc = Dm.pop("router_acc")
    mixed = evaluate([(d, te[d], {k: v[lo:hi] for k, v in Dm.items()}) for d, lo, hi in zip(datasets, offs[:-1], offs[1:])], a.n_boot)
    mixed["router_dataset_accuracy"] = router_acc
    rep = dict(model=a.model, labels=a.labels, runs=spec, n_test={d: len(te[d]) for d in datasets}, per_dataset_gates=per, mixed_traffic_gate=mixed)
    tag = "" if a.labels == "official" else "__" + a.labels
    out = ROOT / "experiments/results" / f"operating_point__{a.model}{tag}"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1, default=float))
    lines = [f"# Operating points (PREREG v2e, exploratory): {a.model}, labels={a.labels}",
             "Held-out accuracy (share of queries that read evidence). Rules are budget-free; nothing is tuned on held-out data."]
    lines += table(per, datasets, "Per-dataset gates (fit on each dataset's dev split)")
    lines += table(mixed, datasets, "Mixed traffic (one gate fit on the pooled dev splits, no dataset identity)")
    pathlib.Path(str(out) + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

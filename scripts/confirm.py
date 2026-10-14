"""Confirmatory analysis (PREREG v1): fit every gate ONCE on the dev split, apply unchanged to the confirm split.

  python scripts/confirm.py --train_run experiments/runs/J1 --test_run experiments/runs/J4 \
      --model Qwen3-VL-4B-Instruct [--labels strict] [--transfer]
Primary endpoint: pooled paired dnAUC(benefit_post - error_post), 95% image-cluster bootstrap CI.
Writes <test_run>/confirm__<model>[__strict].{json,md}.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

FEATS = dict(pre=A.GROUPS["free"], post=A.GROUPS["free"] + A.GROUPS["qpp"])


def fit_predict(Xtr, ytr, Xte, kind):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    m.fit(np.nan_to_num(Xtr), ytr)
    p, cls = m.predict_proba(np.nan_to_num(Xte)), list(m.classes_)
    if kind == "binary":
        return p[:, cls.index(1)]
    return (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)


def probe(Htr, ytr, Hte, kind):
    """Hidden-state probe on the closed-book pass (a learned pre-search router): PCA-64 + logistic, fit on train only."""
    from sklearn.decomposition import PCA
    m = make_pipeline(StandardScaler(), PCA(n_components=64, random_state=0), LogisticRegression(C=0.1, max_iter=3000))
    m.fit(Htr, ytr)
    p, cls = m.predict_proba(Hte), list(m.classes_)
    return p[:, cls.index(1)] if kind == "binary" else p[:, cls.index(1)] - p[:, cls.index(-1)]


def scores(tr, te, Htr=None, Hte=None):
    """Gate scores on `te` from models fit on `tr` (same features / labels as the PREREG 2x2)."""
    S = {}
    for fs, cols in FEATS.items():
        S[f"error_{fs}"] = fit_predict(tr[cols].to_numpy(), (1 - tr.c_ne).to_numpy(), te[cols].to_numpy(), "binary")
        S[f"benefit_{fs}"] = fit_predict(tr[cols].to_numpy(), tr.u.to_numpy(), te[cols].to_numpy(), "benefit")
    hit = fit_predict(tr[FEATS["post"]].to_numpy(), tr.lab_gold_in_top5.to_numpy(), te[FEATS["post"]].to_numpy(), "binary")
    S["factorised_post"] = S["error_post"] * hit
    for s, o in A.ORIENT.items():
        if s in te and te[s].notna().mean() > 0.95 and s in FEATS["post"]:
            S[f"signal_{s}"] = o * te[s].fillna(tr[s].median()).to_numpy()
    S["random"] = np.random.default_rng(0).random(len(te))
    if Htr is not None and Hte is not None:
        S["probe_error"] = probe(Htr, (1 - tr.c_ne).to_numpy(), Hte, "binary")
        S["probe_benefit"] = probe(Htr, tr.u.to_numpy(), Hte, "benefit")
    S["oracle"] = te.u.to_numpy().astype(float)
    return S


def summarize(te, S, unit="tok"):
    c_ne, c_ev = te.c_ne.to_numpy(), te.c_ev.to_numpy()
    target = c_ev.mean() - 0.005
    out = {}
    for k, s in S.items():
        base = U.base_cost(te, ["free"], unit).to_numpy()
        acc, cost = gate.curve(s, c_ne, c_ev, base, te[f"{unit}_ev"].to_numpy())
        ok = np.where(acc >= target - 1e-12)[0]
        gain = c_ev.mean() - c_ne.mean()
        out[k] = dict(nauc=gate.nauc(acc), pauc70=gate.nauc(acc, lo=0.7), acc_at_50=float(acc[50]),
                      gain_recovered_at_50=float((acc[50] - c_ne.mean()) / gain) if gain > 0 else np.nan,
                      read_rate_at_always_acc=float(gate.GRID[ok[0]]) if len(ok) else np.nan,
                      rel_token_cost_at_always_acc=gate.cost_at_accuracy(acc, cost, target) / te[f"{unit}_ev"].mean(),
                      help_auroc=gate.auroc(s, te.u == 1), harm_auroc=gate.auroc(s, te.u == -1),
                      curve_acc=acc.round(5).tolist(), curve_rel_tokens=(cost / te[f"{unit}_ev"].mean()).round(5).tolist())
    return out


def paired(parts, a, b, alpha=0.05, n_boot=2000, lo_rate=0.0):
    """Pooled (mean over datasets) dnAUC(a - b), resampling image groups within each dataset (stratified)."""
    allg = np.concatenate([te.group.to_numpy() for te, _ in parts])
    strata = np.concatenate([np.full(len(te), j) for j, (te, _) in enumerate(parts)])
    offs = np.cumsum([0] + [len(te) for te, _ in parts])

    def stat(idx):
        v = []
        for (te, S), lo, hi in zip(parts, offs[:-1], offs[1:]):
            i = idx[(idx >= lo) & (idx < hi)] - lo
            z = np.zeros(len(i))
            cn, ce = te.c_ne.to_numpy()[i], te.c_ev.to_numpy()[i]
            v.append(gate.nauc(gate.curve(S[a][i], cn, ce, z, z)[0], lo=lo_rate)
                     - gate.nauc(gate.curve(S[b][i], cn, ce, z, z)[0], lo=lo_rate))
        return float(np.mean(v))
    return gate.cluster_bootstrap(allg, stat, n_boot=n_boot, alpha=alpha, strata=strata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_run", required=True)
    ap.add_argument("--test_run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="evqa,infoseek")
    ap.add_argument("--train_split", default="dev")
    ap.add_argument("--test_split", default="confirm")
    ap.add_argument("--labels", default="official", choices=["official", "strict", "relaxed"])
    ap.add_argument("--transfer", action="store_true", help="also fit on one dataset, test on the other")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    dss = a.datasets.split(",")
    load = lambda run, ds, sp: A.load(run, a.data, ds, a.model, sp, a.labels)
    tr = {d: load(a.train_run, d, a.train_split) for d in dss}
    te = {d: load(a.test_run, d, a.test_split) for d in dss}
    parts, rep = [], {}
    Htr = {d: U.load_hidden(a.train_run, d, a.model) for d in dss}
    Hte = {d: U.load_hidden(a.test_run, d, a.model) for d in dss}
    layer = lambda H: sorted(H)[len(H) // 2] if H else None  # middle captured layer (3L/4), fixed a priori
    for d in dss:
        L_ = layer(Htr[d]) if Htr[d] and Hte[d] else None
        hx = (Htr[d][L_].reindex(tr[d].index).to_numpy(), Hte[d][L_].reindex(te[d].index).to_numpy()) if L_ else (None, None)
        if hx[0] is not None and (np.isnan(hx[0]).any() or np.isnan(hx[1]).any()):
            hx = (None, None)
        S = scores(tr[d], te[d], *hx)
        parts.append((te[d], S))
        keep = ["image_id", "c_ne", "c_ev", "u", "ne_text", "ev_text"] + [c for c in te[d] if c.startswith(("tok_", "gpu_"))]
        per_q = te[d][[c for c in keep if c in te[d]]].copy()
        for k, v in S.items():
            per_q[f"score_{k}"] = v
        if a.labels == "official":
            per_q.to_csv(pathlib.Path(a.test_run) / f"gate_scores__{d}__{a.model}__{a.test_split}.csv")
        rep[d] = dict(n_train=len(tr[d]), n_test=len(te[d]), images_test=int(te[d].image_id.nunique()),
                      never_rel_tokens=float(te[d].tok_ne.mean() / te[d].tok_ev.mean()),
                      quadrants={k: gate.cluster_bootstrap(te[d].group.to_numpy(), lambda i, k=k: gate.outcome_table(
                          te[d].c_ne.to_numpy()[i], te[d].c_ev.to_numpy()[i])[k], n_boot=a.n_boot)
                          for k in ["help", "harm", "acc_never", "acc_always", "acc_oracle_gate"]},
                      policies=summarize(te[d], S))
    comps = [("benefit_post", "error_post"), ("benefit_pre", "error_pre"), ("benefit_post", "factorised_post"),
             ("benefit_post", "error_pre"), ("factorised_post", "error_post"), ("benefit_post", "signal_qpp_nqc"),
             ("signal_qpp_nqc", "error_post"), ("probe_benefit", "probe_error"), ("benefit_post", "random")]
    comps = [(x, y) for x, y in comps if all(x in S_ and y in S_ for _, S_ in parts)]
    rep["pooled_dnauc_95ci"] = {f"{x} - {y}": paired(parts, x, y, n_boot=a.n_boot) for x, y in comps}
    rep["pooled_pauc70_95ci_exploratory"] = {f"{x} - {y}": paired(parts, x, y, n_boot=a.n_boot, lo_rate=0.7)
                                             for x, y in comps[:2] + comps[5:7]}
    rep["per_dataset_dnauc_95ci"] = {d: {f"{x} - {y}": paired([p], x, y, n_boot=a.n_boot) for x, y in comps[:1]}
                                     for d, p in zip(dss, parts)}
    if a.transfer and len(dss) == 2:
        rep["transfer"] = {}
        for src, dst in [(dss[0], dss[1]), (dss[1], dss[0])]:
            S = scores(tr[src], te[dst])
            rep["transfer"][f"{src}->{dst}"] = {k: v["nauc"] for k, v in summarize(te[dst], S).items()
                                                if k in ("benefit_post", "error_post", "factorised_post", "oracle")}
    pt, lo, hi = rep["pooled_dnauc_95ci"]["benefit_post - error_post"]
    rep["primary"] = dict(point_pt=pt * 100, ci95_pt=[lo * 100, hi * 100], confirmed=bool(pt > 0 and lo > 0))
    tag = "" if a.labels == "official" else "__" + a.labels
    out = pathlib.Path(a.test_run) / f"confirm__{a.model}{tag}"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1, default=float))
    lines = [f"# Confirmatory: {a.model} (fit on {a.train_split}, test on {a.test_split}; labels={a.labels})\n",
             f"**Primary** dnAUC(benefit_post - error_post) = {pt*100:+.2f} pt, 95% CI [{lo*100:+.2f}, {hi*100:+.2f}] -> "
             f"{'CONFIRMED' if rep['primary']['confirmed'] else 'NOT CONFIRMED'}\n",
             "| comparison (pooled) | point | 95% lo | 95% hi |\n|---|---|---|---|"]
    lines += [f"| {k} | {v[0]*100:+.2f} | {v[1]*100:+.2f} | {v[2]*100:+.2f} |" for k, v in rep["pooled_dnauc_95ci"].items()]
    lines.append("\n| exploratory: partial AUC, 70-100% reads | point | 95% lo | 95% hi |\n|---|---|---|---|")
    lines += [f"| {k} | {v[0]*100:+.2f} | {v[1]*100:+.2f} | {v[2]*100:+.2f} |" for k, v in rep["pooled_pauc70_95ci_exploratory"].items()]
    for d in dss:
        r = rep[d]
        q = r["quadrants"]
        lines.append(f"\n## {d} (train {r['n_train']}, test {r['n_test']}, images {r['images_test']})\n")
        lines.append("never {:.1%} · always {:.1%} · oracle gate {:.1%} · help {:.1%} · harm {:.1%}\n".format(
            q["acc_never"][0], q["acc_always"][0], q["acc_oracle_gate"][0], q["help"][0], q["harm"][0]))
        lines.append("| policy | nAUC | pAUC70 | acc@50% reads | gain rec.@50% | reads for always-acc−0.5 | rel. tokens at always-acc−0.5 | help AUROC | harm AUROC |\n|---|---|---|---|---|---|---|---|---|")
        for k, v in r["policies"].items():
            lines.append(f"| {k} | {v['nauc']:.4f} | {v['pauc70']:.4f} | {v['acc_at_50']:.3f} | {v['gain_recovered_at_50']:.2f} | "
                         f"{v['read_rate_at_always_acc']:.2f} | {v['rel_token_cost_at_always_acc']:.2f} | {v['help_auroc']:.3f} | {v['harm_auroc']:.3f} |")
    if "transfer" in rep:
        lines.append("\n## Cross-dataset transfer (nAUC)\n")
        lines += [f"- {k}: " + ", ".join(f"{p} {v:.4f}" for p, v in t.items()) for k, t in rep["transfer"].items()]
    pathlib.Path(str(out) + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

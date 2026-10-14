"""Gate-1 / confirmatory analysis for one model over one or more datasets.

  python scripts/analyze.py --run experiments/runs/J1 --model Qwen3-VL-4B-Instruct --datasets evqa,infoseek --split dev

Writes <run>/analysis__<model>__<split>.json and .md: retrieval quadrants, per-signal AUROC for error/help/harm,
the 2x2 {error, benefit label} x {pre, post-search features} gates (OOF, GroupKFold by image), factorised
P(wrong)*P(hit), oracle/never/always, nAUC over retrieval rate, total-token cost at always-retrieve accuracy,
and paired dnAUC (benefit - error) with image-cluster bootstrap CIs (80% for the pilot, per PREREG).
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from vgate import gate, uncertainty as U  # noqa: E402

# +1: larger value = more uncertain (-> retrieve); for QPP +1 = larger value = retrieval looks better.
ORIENT = dict(ent_mean=1, ent_max=1, ent_first=1, lp_min=-1, lp_mean=-1, seq_lp=-1, sample_agree=-1,
              sample_distinct=1, ent_agree=-1, ent_distinct=1, p_true=-1, para_agree=-1,
              qpp_top1_z=1, qpp_margin_z=1, qpp_nqc=1, qpp_entropy=-1)
GROUPS = dict(free=U.FREE, sample=U.SAMPLE, entity=U.ENTITY, ptrue=U.PTRUE, para=U.PARA, qpp=U.QPP)


def load(run, data, dataset, model, split, labels="official"):
    recs = [json.loads(l) for l in open(pathlib.Path(data) / "processed" / f"{dataset}.jsonl")]
    keep = set((pathlib.Path(data) / "splits" / f"{dataset}_{split}.txt").read_text().split())
    recs = [r for r in recs if r["qid"] in keep]
    df = U.build(run, dataset, model, recs)
    sc = pd.read_csv(pathlib.Path(run) / f"scores__{dataset}__{model}.csv", index_col="qid")
    sc.index = sc.index.astype(str)  # OK-VQA question ids are numeric strings
    df = df.join(sc, how="inner")
    lab = pd.DataFrame([{k: r[k] for k in ("qid", "lab_gold_in_top1", "lab_gold_in_top5")} for r in recs])
    df = df.join(lab.set_index("qid"))
    if labels == "strict" and "cs_ne" in df:  # numeric-strict sensitivity labels (E-VQA only; see score.py)
        for c in [c for c in df if c.startswith("cs_")]:
            df["c_" + c[3:]] = df[c]
    if labels == "relaxed" and "cr_ne" in df:  # containment-relaxed sensitivity labels (OK-VQA only)
        for c in [c for c in df if c.startswith("cr_")]:
            df["c_" + c[3:]] = df[c]
    df = df[df.c_ev.notna() & df.c_ne.notna()].copy()
    df["u"] = (df.c_ev - df.c_ne).astype(int)
    df["group"] = dataset + "/" + df.image_id.astype(str)
    return df


def available(df, groups):
    return [g for g in groups if all(c in df and df[c].notna().mean() > 0.95 for c in GROUPS[g])]


def policies(df, hidden=None, pre_groups=None):
    """Out-of-fold retrieve-scores for every policy, plus the feature groups each one pays for."""
    pre = available(df, pre_groups or ["free", "sample", "entity", "ptrue", "para"])
    post = pre + ["qpp"]
    cols = lambda gs: [c for g in gs for c in GROUPS[g]]
    y_err, u, grp = (1 - df.c_ne).to_numpy(), df.u.to_numpy(), df.group.to_numpy()
    P = {}
    for name, gs in [("pre", pre), ("post", post), ("free", ["free"])]:
        X = df[cols(gs)].to_numpy()
        P[f"error_{name}"] = (gate.oof_scores(X, y_err, grp), gs)
        P[f"benefit_{name}"] = (gate.oof_scores(X, u, grp, kind="benefit"), gs)
    p_hit = gate.oof_scores(df[cols(post)].to_numpy(), df.lab_gold_in_top5.to_numpy(), grp)
    P["factorised_post"] = (P["error_post"][0] * p_hit, post)
    for s, o in ORIENT.items():  # single-signal thresholds (Moskvoretskii-style) and pure-QPP gates
        if s in df and df[s].notna().mean() > 0.95:
            g = next(k for k, v in GROUPS.items() if s in v)
            P[f"signal_{s}"] = (o * df[s].fillna(df[s].median()).to_numpy(), ["free", g] if g != "free" else ["free"])
    for layer, H in (hidden or {}).items():  # closed-book hidden-state probes (learned pre-search routers), free cost
        Xh = H.reindex(df.index).to_numpy()
        if np.isnan(Xh).any():
            continue
        P[f"probe_error_L{layer}"] = (gate.oof_scores(Xh, y_err, grp, pca=64, C=0.1), ["free"])
        P[f"probe_benefit_L{layer}"] = (gate.oof_scores(Xh, u, grp, kind="benefit", pca=64, C=0.1), ["free"])
    P["oracle"] = (u.astype(float), ["free"])
    return P, pre, post


def evaluate(df, P, unit="tok"):
    c_ne, c_ev = df.c_ne.to_numpy(), df.c_ev.to_numpy()
    ev_cost = df[f"{unit}_ev"].to_numpy()
    always_cost = ev_cost.mean()
    target = c_ev.mean() - 0.005
    res = {}
    for name, (s, gs) in P.items():
        base = U.base_cost(df, gs, unit).to_numpy()
        acc, cost = gate.curve(s, c_ne, c_ev, base, ev_cost)
        ok = np.where(acc >= target - 1e-12)[0]
        res[name] = dict(nauc=gate.nauc(acc), acc_at_50=float(acc[50]),
                         search_rate_at_always_acc=float(gate.GRID[ok[0]]) if len(ok) else np.nan,
                         rel_cost_at_always_acc=gate.cost_at_accuracy(acc, cost, target) / always_cost,
                         best_acc=float(acc.max()), rate_at_best=float(gate.GRID[acc.argmax()]),
                         curve_acc=acc.round(5).tolist(), curve_cost=(cost / always_cost).round(5).tolist())
    res["never"] = dict(acc=float(c_ne.mean()), rel_cost=float(df[f"{unit}_ne"].mean() / always_cost))
    res["always"] = dict(acc=float(c_ev.mean()), rel_cost=1.0)
    return res


def paired_dnauc(dfs, Ps, a, b, alpha=0.2, n_boot=2000):
    """dnAUC(a - b) pooled over datasets (mean), resampling image groups within each dataset."""
    parts = []
    for df, P in zip(dfs, Ps):
        parts.append((df.c_ne.to_numpy(), df.c_ev.to_numpy(), P[a][0], P[b][0], df.group.to_numpy()))
    allg = np.concatenate([p[4] for p in parts])
    offs = np.cumsum([0] + [len(p[0]) for p in parts])

    def stat(idx):
        vals = []
        for (cn, ce, sa, sb, _), lo, hi in zip(parts, offs[:-1], offs[1:]):
            i = idx[(idx >= lo) & (idx < hi)] - lo
            z = np.zeros(len(i))
            da = gate.nauc(gate.curve(sa[i], cn[i], ce[i], z, z)[0])
            db = gate.nauc(gate.curve(sb[i], cn[i], ce[i], z, z)[0])
            vals.append(da - db)
        return float(np.mean(vals))
    return gate.cluster_bootstrap(allg, stat, n_boot=n_boot, alpha=alpha)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="evqa,infoseek")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--data", default="data")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--labels", default="official", choices=["official", "strict"])
    ap.add_argument("--pre_groups", default="", help="restrict pre-search feature groups, e.g. 'free' (PREREG v1 pruning)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    a = ap.parse_args()
    report, dfs, Ps = {}, [], []
    for ds in a.datasets.split(","):
        df = load(a.run, a.data, ds, a.model, a.split, a.labels)
        P, pre, post = policies(df, U.load_hidden(a.run, ds, a.model), a.pre_groups.split(",") if a.pre_groups else None)
        dfs.append(df); Ps.append(P)
        keep = ["image_id", "c_ne", "c_ev", "u", "ne_text", "ev_text"] + [c for c in df if c.startswith(("tok_", "gpu_"))]
        per_q = df[[c for c in keep if c in df]].copy()
        for name, (sc, _) in P.items():
            per_q[f"score_{name}"] = sc
        if a.labels == "official" and not a.tag:
            per_q.to_csv(pathlib.Path(a.run) / f"gate_scores__{ds}__{a.model}__{a.split}.csv")
        grp = df.group.to_numpy()
        quad = {k: gate.cluster_bootstrap(grp, lambda i, k=k: gate.outcome_table(df.c_ne.to_numpy()[i],
                                                                                df.c_ev.to_numpy()[i])[k],
                                          n_boot=a.n_boot, alpha=a.alpha)
                for k in ["help", "harm", "both_wrong", "both_right", "acc_never", "acc_always", "acc_oracle_gate"]}
        if "c_oracle" in df:
            quad["acc_oracle_evidence"] = (float(df.c_oracle.mean()),) * 3
        aur = {}
        for s, o in ORIENT.items():
            if s in df and df[s].notna().mean() > 0.95:
                x = o * df[s].fillna(df[s].median()).to_numpy()
                aur[s] = dict(error=gate.auroc(x, df.c_ne == 0), help=gate.auroc(x, df.u == 1),
                              harm=gate.auroc(x, df.u == -1), help_vs_harm=gate.auroc(x[df.u != 0], df.u[df.u != 0] == 1))
        report[ds] = dict(n=len(df), n_images=int(df.image_id.nunique()), feature_groups_pre=pre,
                          quadrants=quad, auroc=aur, policies=evaluate(df, P),
                          policies_gpu=evaluate(df, P, unit="gpu") if "gpu_ev" in df else None)
    pooled = {}
    for a_, b_ in [("benefit_pre", "error_pre"), ("benefit_post", "error_post"), ("benefit_free", "error_free"),
                   ("benefit_post", "factorised_post"), ("error_post", "error_pre"), ("benefit_post", "benefit_pre")]:
        if all(a_ in P and b_ in P for P in Ps):
            pooled[f"{a_} - {b_}"] = paired_dnauc(dfs, Ps, a_, b_, alpha=a.alpha, n_boot=a.n_boot)
    report["pooled_dnauc"] = pooled
    out = pathlib.Path(a.run) / f"analysis__{a.model}__{a.split}{'' if a.labels == 'official' else '__' + a.labels}{a.tag}"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(report, indent=1, default=float))
    lines = [f"# Analysis: {a.model} / {a.split}\n"]
    for ds in a.datasets.split(","):
        r = report[ds]
        lines.append(f"## {ds} (n={r['n']}, images={r['n_images']}, pre-search groups={r['feature_groups_pre']})\n")
        lines.append("| quantity | point | lo | hi |\n|---|---|---|---|")
        lines += [f"| {k} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} |" for k, v in r["quadrants"].items()]
        lines.append("\n| signal | AUROC error | help | harm | help-vs-harm |\n|---|---|---|---|---|")
        lines += [f"| {s} | {v['error']:.3f} | {v['help']:.3f} | {v['harm']:.3f} | {v['help_vs_harm']:.3f} |"
                  for s, v in r["auroc"].items()]
        lines.append("\n| policy | nAUC | acc@50% | rel. token cost at always-acc−0.5pt | best acc (rate) |\n|---|---|---|---|---|")
        for k, v in r["policies"].items():
            if "nauc" in v:
                lines.append(f"| {k} | {v['nauc']:.4f} | {v['acc_at_50']:.4f} | {v['rel_cost_at_always_acc']:.3f} | "
                             f"{v['best_acc']:.4f} ({v['rate_at_best']:.2f}) |")
        lines.append(f"| never | acc {r['policies']['never']['acc']:.4f} | | rel cost {r['policies']['never']['rel_cost']:.3f} | |")
        lines.append(f"| always | acc {r['policies']['always']['acc']:.4f} | | rel cost 1.000 | |\n")
    lines.append("## Pooled paired dnAUC (image-cluster bootstrap)\n\n| comparison | point | lo | hi |\n|---|---|---|---|")
    lines += [f"| {k} | {v[0]*100:+.2f} pt | {v[1]*100:+.2f} | {v[2]*100:+.2f} |" for k, v in pooled.items()]
    pathlib.Path(str(out) + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

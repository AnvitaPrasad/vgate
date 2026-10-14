"""Secondary analyses the paper quotes, consolidated so the pipeline is reproducible from run outputs:

  experiments/results/signal_auroc_confirm.csv     per-signal error/help/harm AUROC on held-out data (both models)
  experiments/runs/J4/cascade__<M1>.json           VGate-pre -> VGate-post cascade at pre-specified budgets 0.9, 0.5
  experiments/runs/J3/c2_summary.json              C2 (Thinking) criteria: unique help vs noise floor, reliability
  experiments/runs/J3/c2_breaks_given_right.json   thinking/retrieval break rates among NE-correct answers
  experiments/results/pass_costs.csv               measured per-pass tokens and GPU-seconds
  experiments/results/label_efficiency.csv         held-out nAUC of VGate-post fit on n labelled dev queries (exploratory)

  python scripts/extra_analyses.py      (needs scores__*.csv from scripts/score.py and gate_scores from confirm.py)
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
import analyze as A  # noqa: E402
from vgate import gate  # noqa: E402

RUNS = ROOT / "experiments/runs"
M1, M2, MT = "Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct", "Qwen3-VL-4B-Thinking"
DS = ("evqa", "infoseek")


def signal_auroc():
    rows = []
    for run, model in [("J4", M1), ("J2", M2)]:
        for d in DS:
            df = A.load(RUNS / run, ROOT / "data", d, model, "confirm")
            for s in ["ent_mean", "lp_min", "seq_lp", "qpp_top1_z", "qpp_margin_z", "qpp_nqc", "qpp_entropy"]:
                x = A.ORIENT[s] * df[s].fillna(df[s].median()).to_numpy()
                rows.append(dict(model=model.split("-")[0] + model.split("-")[1], dataset=d, signal=s,
                                 err=gate.auroc(x, df.c_ne == 0), help=gate.auroc(x, df.u == 1), harm=gate.auroc(x, df.u == -1)))
    pd.DataFrame(rows).to_csv(ROOT / "experiments/results/signal_auroc_confirm.csv", index=False)


def cascade():
    out = {}
    for d in DS:
        g = pd.read_csv(RUNS / "J4" / f"gate_scores__{d}__{M1}__confirm.csv", index_col="qid")
        for b in (0.5, 0.9):  # budgets fixed before looking (PREREG outcome log, 2026-10-01 02:40)
            search = g.score_probe_benefit >= np.quantile(g.score_probe_benefit, 1 - b)
            for key, col in [("", "score_benefit_post"), ("_full", "score_vgate_full")]:  # read gate: VGate-post / VGate-full
                if col not in g:
                    continue
                read = search & (g[col] >= np.quantile(g[col], 1 - b))
                out[f"{d}_{b}{key}"] = dict(searches=float(search.mean()), reads=float(read.mean()),
                                            acc=float(np.where(read, g.c_ev, g.c_ne).mean()),
                                            rel_tokens=float((g.tok_ne + np.where(read, g.tok_ev, 0)).mean() / g.tok_ev.mean()),
                                            acc_always=float(g.c_ev.mean()), acc_never=float(g.c_ne.mean()))
    (RUNS / "J4" / f"cascade__{M1}.json").write_text(json.dumps(out, indent=1))


def c2():
    summ, brk = {}, {}
    for d in DS:
        base = pd.read_csv(RUNS / "J1" / f"scores__{d}__{M1}.csv", index_col="qid")[["c_ne", "c_ev"]]
        th = pd.read_csv(RUNS / "J3" / f"scores__{d}__{MT}.csv", index_col="qid")
        ns = pd.read_csv(RUNS / "J3" / f"scores__{d}__{M1}.csv", index_col="qid")
        df = base.join(th[["c_think", "c_think_s2"]], how="inner").join(ns[["c_ne_tset"]], how="inner")
        img = {r["qid"]: r["image_id"] for r in map(json.loads, open(ROOT / "data/processed" / f"{d}.jsonl"))}
        grp = np.array([img[q] for q in df.index])
        cn, ce, ct, cf = (df[c].to_numpy() for c in ["c_ne", "c_ev", "c_think", "c_ne_tset"])
        ut = lambda i: float(np.mean((ct[i] == 1) & (cn[i] == 0) & (ce[i] == 0)))
        uf = lambda i: float(np.mean((cf[i] == 1) & (cn[i] == 0) & (ce[i] == 0)))
        s2 = df.dropna(subset=["c_think_s2"])
        summ[d] = dict(n=len(df), acc_ne=float(cn.mean()), acc_ev=float(ce.mean()), acc_think=float(ct.mean()),
                       acc_ne_tset=float(cf.mean()), unique_help_think=ut(np.arange(len(df))),
                       unique_help_noise=uf(np.arange(len(df))),
                       c2a_diff=gate.cluster_bootstrap(grp, lambda i: ut(i) - uf(i), n_boot=2000, alpha=0.2),
                       think_breaks_correct_ne=float(np.mean((cn == 1) & (ct == 0))),
                       think_label_agreement_s2=float((s2.c_think == s2.c_think_s2).mean()),
                       think_acc_on_ne_wrong=float(ct[cn == 0].mean()), ev_acc_on_ne_wrong=float(ce[cn == 0].mean()))
        r = df[df.c_ne == 1]
        brk[d] = dict(think_breaks_given_right=float((r.c_think == 0).mean()),
                      ev_breaks_given_right=float((r.c_ev == 0).mean()), n_right=len(r))
    (RUNS / "J3" / "c2_summary.json").write_text(json.dumps(summ, indent=1))
    (RUNS / "J3" / "c2_breaks_given_right.json").write_text(json.dumps(brk, indent=1))


def pass_costs():
    rows = []
    spec = [("J4", M1, c) for c in ("ne", "ev", "oracle")] + [("J2", M2, c) for c in ("ne", "ev", "oracle")]
    spec += [("J1", M1, c) for c in ("ne_sample", "ent_sample", "ptrue", "para")] + [("J3", MT, "think")]
    for run, model, c in spec:
        for d in DS:
            r = [json.loads(l) for l in open(RUNS / run / f"{d}__{model}__{c}.jsonl")]
            rows.append(dict(model=model, dataset=d, pass_=c, n=len(r), prompt_tok=np.mean([x["n_prompt"] for x in r]),
                             gen_tok=np.mean([x["n_gen"] for x in r]), gpu_s=np.mean([x["gpu_s"] for x in r])))
    pd.DataFrame(rows).round(2).to_csv(ROOT / "experiments/results/pass_costs.csv", index=False)


def label_efficiency(sizes=(50, 100, 200, 400), seeds=10):
    """How many labelled queries does a gate need? VGate-post fit on random image-group subsamples of dev,
    scored on the held-out split (exploratory; the error gate fit on all dev queries is the reference)."""
    import confirm as C
    rows = []
    for d in DS:
        tr, te = A.load(RUNS / "J1", ROOT / "data", d, M1, "dev"), A.load(RUNS / "J4", ROOT / "data", d, M1, "confirm")
        cols, z = C.FEATS["post"], np.zeros(len(te))
        na = lambda sc: gate.nauc(gate.curve(sc, te.c_ne.to_numpy(), te.c_ev.to_numpy(), z, z)[0])
        fit = lambda sub, y, kind: C.fit_predict(sub[cols].to_numpy(), y, te[cols].to_numpy(), kind)
        rows.append(dict(dataset=d, gate="error_post", n=len(tr), seeds=1, nauc=na(fit(tr, (1 - tr.c_ne).to_numpy(), "binary")), sd=0.0))
        rows.append(dict(dataset=d, gate="benefit_post", n=len(tr), seeds=1, nauc=na(fit(tr, tr.u.to_numpy(), "benefit")), sd=0.0))
        groups = tr.group.unique()
        for n in sizes:
            v = []
            for seed in range(seeds):
                keep, tot = [], 0
                for g in np.random.default_rng(seed).permutation(groups):
                    if tot >= n:
                        break
                    keep.append(g); tot += int((tr.group == g).sum())
                sub = tr[tr.group.isin(keep)]
                if len(np.unique(sub.u)) == 3:
                    v.append(na(fit(sub, sub.u.to_numpy(), "benefit")))
            rows.append(dict(dataset=d, gate="benefit_post", n=n, seeds=len(v), nauc=float(np.mean(v)), sd=float(np.std(v))))
    pd.DataFrame(rows).round(5).to_csv(ROOT / "experiments/results/label_efficiency.csv", index=False)


if __name__ == "__main__":
    signal_auroc(); cascade(); c2(); pass_costs(); label_efficiency()
    print("extra analyses written")

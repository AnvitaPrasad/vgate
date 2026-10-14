"""PREREG v3: passage-level benefit selection ("which passage?").

Per question there are six candidate answers: a_0 (closed-book) and a_k from reading retrieved passage k alone
(conditions ev1_1..ev1_5). VGate-pick scores each passage by its predicted benefit u_k = c_k - c_0 and answers with the
best passage if its score is positive, else with a_0. A correctness-target counterpart and training-free baselines are
evaluated on the same candidates.

  pilot (dev, grouped 5-fold OOF):
    python scripts/passage_select.py --run experiments/runs/J10 --logged evqa=J1,infoseek=J1,okvqa=J6 --split dev
  held-out (fit on dev, apply to confirm):
    python scripts/passage_select.py --run <confirm run> --train_run experiments/runs/J10 --logged ... --logged_train ... --split confirm
Writes <run>/passage_select__<model>__<split>.{json,md}. Needs scores__<dataset>__<model>.csv in the run folders
(scripts/score.py scores ev1_k and ev_keep).
"""
import argparse
import json
import pathlib
import re
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
from vgate import gate, uncertainty as U  # noqa: E402

K = 5
ABSTAIN = re.compile(r"\bnot (specified|available|answerable|determinable|mentioned|provided|given|stated|known|clear)"
                     r"|\bnone\b|\bunknown\b|cannot|can't", re.I)
ANS = ["ent_mean", "ent_max", "ent_first", "lp_min", "lp_mean", "seq_lp", "n_gen"]          # statistics of answer a_k
PVF = ["p_yes", "pv_rank", "pv_gap"]                                                         # verification pass
OTH = ["abstain", "agree0", "n_agree", "rank", "retr_z"] + ["ne_" + f for f in U.FREE]       # no gold / hit / oracle field
FEATS = ANS + PVF + OTH


def stats(row):
    lp, en = np.array(row["tok_logprobs"] or [0.0]), np.array(row["tok_entropy"] or [0.0])
    return dict(ent_mean=en.mean(), ent_max=en.max(), ent_first=en[0], lp_min=lp.min(), lp_mean=lp.mean(), seq_lp=lp.sum(),
                n_gen=row["n_gen"])


def candidates(run, logged, data, d, model, split):
    """One row per (question, passage k=1..5) with features, labels c_k / u_k and costs; plus the per-question frame."""
    run = pathlib.Path(run)
    q = A.load(logged, data, d, model, split)  # logged closed-book / top-5 passes: c_ne, c_ev, free features, tokens, group
    recs = {r["qid"]: r for r in map(json.loads, open(pathlib.Path(data) / "processed" / f"{d}.jsonl")) if r["qid"] in q.index}
    sc = pd.read_csv(run / f"scores__{d}__{model}.csv", index_col="qid"); sc.index = sc.index.astype(str)
    ev1 = [U.read(run, d, model, f"ev1_{k}") for k in range(1, K + 1)]
    pv = [U.read(run, d, model, f"pv_{k}") for k in range(1, K + 1)]
    keep = U.read(run, d, model, "ev_keep")
    have_pv = all(len(p) > 0 for p in pv)
    q = q.loc[[i for i in q.index if all(i in e for e in ev1) and i in sc.index and (not have_pv or all(i in p for p in pv))]].copy()
    rows = []
    for i in q.index:
        r, a0 = recs[i], U.norm(q.ne_text[i])
        s20 = np.asarray(r["retr_scores"], float)
        mu, sd = s20.mean(), s20.std() + 1e-9
        ans = [U.norm(ev1[k][i]["text"]) for k in range(K)]
        py = np.array([pv[k][i]["p_yes"] for k in range(K)]) if have_pv else np.zeros(K)
        order = (-py).argsort().argsort()
        for k in range(K):
            e = ev1[k][i]
            row = dict(qid=i, k=k + 1, **stats(e), abstain=float(bool(ABSTAIN.search(e["text"] or "")) or not (e["text"] or "").strip()),
                       agree0=float(ans[k] == a0), n_agree=float(sum(ans[j] == ans[k] for j in range(K) if j != k)),
                       p_yes=float(py[k]), pv_rank=float(order[k]), pv_gap=float(py[k] - np.delete(py, k).max()),
                       rank=float(k + 1), retr_z=float((s20[k] - mu) / sd), c=int(sc.at[i, f"c_ev1_{k + 1}"]),
                       tok=e["n_prompt"] + e["n_gen"], tok_pv=(pv[k][i]["n_prompt"] + 1) if have_pv else 0, text=e["text"])
            row.update({"ne_" + f: q.at[i, f] for f in U.FREE})
            rows.append(row)
    c = pd.DataFrame(rows)
    c["c0"] = c.qid.map(q.c_ne).astype(int)
    c["u"] = c.c - c.c0
    c["group"] = c.qid.map(q.group)
    q["c_keep"] = [int(sc.at[i, "c_ev_keep"]) if "c_ev_keep" in sc and i in keep else np.nan for i in q.index]
    q["tok_keep"] = [keep[i]["n_prompt"] + keep[i]["n_gen"] if i in keep else np.nan for i in q.index]
    run_ne = U.read(run, d, model, "ne")  # the closed-book pass re-run by this job: should reproduce the logged one
    q["ne_agree"] = [float(U.norm(run_ne[i]["text"]) == U.norm(q.ne_text[i])) if i in run_ne else np.nan for i in q.index]
    return c, q, have_pv


def fit_benefit(X, u):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(np.nan_to_num(X), u)


def benefit_score(m, X):
    p, cls = m.predict_proba(np.nan_to_num(X)), list(m[-1].classes_)
    return (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)


def closed_book_rows(c, q):
    """Candidate k=0 for the correctness-target selector: the closed-book answer with its own token statistics."""
    z = pd.DataFrame(dict(qid=q.index, k=0, ent_mean=q.ent_mean, ent_max=q.ent_max, ent_first=q.ent_first, lp_min=q.lp_min,
                          lp_mean=q.lp_mean, seq_lp=q.seq_lp, n_gen=q.n_gen_ne, abstain=0.0, agree0=1.0,
                          n_agree=c.groupby("qid").agree0.sum().reindex(q.index).to_numpy(), p_yes=0.0, pv_rank=0.0, pv_gap=0.0,
                          rank=0.0, retr_z=0.0, c=q.c_ne.astype(int), group=q.group, is0=1.0))
    for f in U.FREE:
        z["ne_" + f] = q[f].to_numpy()
    return z


def select(c, q, tr_q, te_q, feats):
    """Fit on the questions tr_q, pick for the questions te_q. Returns {policy: Series(qid -> chosen k, 0 = closed-book)}."""
    tr, te = c[c.qid.isin(tr_q)], c[c.qid.isin(te_q)].copy()
    out = {}
    te["s"] = benefit_score(fit_benefit(tr[feats].to_numpy(), tr.u.to_numpy()), te[feats].to_numpy())
    best = te.loc[te.groupby("qid").s.idxmax()]
    out["pick_benefit"] = pd.Series(np.where(best.s > 0, best.k, 0), index=best.qid)
    out["score_benefit"] = pd.Series(best.s.to_numpy(), index=best.qid)
    # correctness target over six candidates
    z = closed_book_rows(c, q)
    allc = pd.concat([c.assign(is0=0.0), z], ignore_index=True)
    f6 = feats + ["is0"]
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(
        np.nan_to_num(allc[allc.qid.isin(tr_q)][f6].to_numpy()), allc[allc.qid.isin(tr_q)].c.to_numpy())
    t6 = allc[allc.qid.isin(te_q)].copy()
    t6["s"] = m.predict_proba(np.nan_to_num(t6[f6].to_numpy()))[:, list(m[-1].classes_).index(1)]
    b6 = t6.loc[t6.groupby("qid").s.idxmax()]
    out["pick_correct"] = pd.Series(b6.k.to_numpy(), index=b6.qid)
    return out


def baselines(c, q, have_pv):
    g = c.set_index(["qid", "k"])
    out = {"never": pd.Series(0, index=q.index), "top1": pd.Series(1, index=q.index)}
    conf = c[c.abstain == 0]
    b = conf.loc[conf.groupby("qid").lp_mean.idxmax()]
    out["confident"] = pd.Series(b.k.to_numpy(), index=b.qid).reindex(q.index).fillna(0).astype(int)
    if have_pv:
        b = c.loc[c.groupby("qid").p_yes.idxmax()]
        out["verify"] = pd.Series(np.where(b.p_yes >= 0.5, b.k, 0), index=b.qid).reindex(q.index).astype(int)
    vote = {}
    for i, grp in conf.assign(a=conf.text.map(U.norm)).groupby("qid"):
        top = grp.a.value_counts()
        vote[i] = int(grp[grp.a == top.index[0]].k.min())  # most common answer; ties -> the best-ranked passage
    out["vote"] = pd.Series(vote).reindex(q.index).fillna(0).astype(int)
    return out, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="folder with the ev1_k / pv_k / ev_keep outputs of --split")
    ap.add_argument("--logged", required=True, help="dataset=RUN,... folders (under experiments/runs) with the logged ne / ev passes of --split")
    ap.add_argument("--train_run", default="", help="fit on this run's dev candidates and apply to --run (held-out mode)")
    ap.add_argument("--logged_train", default="", help="dataset=RUN,... logged dev runs for --train_run")
    ap.add_argument("--model", default="Qwen3-VL-4B-Instruct")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.2, help="0.2 = 80% intervals (pilot); 0.05 for held-out")
    a = ap.parse_args()
    R, data = ROOT / "experiments/runs", ROOT / "data"
    logged = dict(x.split("=") for x in a.logged.split(","))
    logged_tr = dict(x.split("=") for x in a.logged_train.split(",")) if a.logged_train else {}
    rep = {}
    for d, lr in logged.items():
        c, q, have_pv = candidates(a.run, R / lr, data, d, a.model, a.split)
        picks = {}
        variants = {"": FEATS, "__no_pv": [f for f in FEATS if f not in PVF], "__no_ans": [f for f in FEATS if f not in ANS]}
        if not have_pv:
            variants = {"": [f for f in FEATS if f not in PVF]}
        for tag, feats in variants.items():
            if a.train_run:  # held-out: fit on dev candidates of the training run
                ctr, qtr, _ = candidates(a.train_run, R / logged_tr[d], data, d, a.model, "dev")
                both_c, both_q = pd.concat([ctr, c], ignore_index=True), pd.concat([qtr, q])
                res = select(both_c, both_q, set(qtr.index), set(q.index), feats)
            else:           # pilot: grouped out-of-fold on the same split
                res, qs = {}, q.index.to_numpy()
                for tr, te in GroupKFold(n_splits=5).split(qs, groups=q.group.to_numpy()):
                    for k_, v in select(c, q, set(qs[tr]), set(qs[te]), feats).items():
                        res[k_] = pd.concat([res[k_], v]) if k_ in res else v
            for k_, v in res.items():
                picks[k_ + tag] = v.reindex(q.index)
        base, g = baselines(c, q, have_pv)
        picks.update(base)
        ck = lambda sel: np.array([q.c_ne[i] if k == 0 else g.at[(i, int(k)), "c"] for i, k in sel.items()], float)
        tok_all = c.groupby("qid").tok.sum().reindex(q.index).to_numpy() + q.tok_ne.to_numpy()
        tok_pv = c.groupby("qid").tok_pv.sum().reindex(q.index).to_numpy()
        acc = {"always": q.c_ev.to_numpy().astype(float), "oracle6": np.maximum(q.c_ne.to_numpy(), c.groupby("qid").c.max().reindex(q.index).to_numpy()).astype(float)}
        tok = {"never": float(q.tok_ne.mean()), "always": float(q.tok_ev.mean()), "oracle6": float(tok_all.mean())}
        tok_of = lambda sel: np.array([0 if k == 0 else g.at[(i, int(k)), "tok"] for i, k in sel.items()], float)
        for k_, sel in picks.items():
            if k_.startswith("score_"):
                continue
            acc[k_] = ck(sel)
            if k_ == "never":
                continue
            if k_ == "top1":        # closed-book pass + the first passage
                tok[k_] = float((q.tok_ne.to_numpy() + tok_of(sel)).mean())
            elif k_ == "verify":    # closed-book pass + five verification passes + the one passage that is read
                tok[k_] = float((q.tok_ne.to_numpy() + tok_pv + tok_of(sel)).mean())
            elif k_ in ("confident", "vote") or k_.endswith("__no_pv") or not have_pv:  # closed-book pass + five reads
                tok[k_] = float(tok_all.mean())
            else:                   # closed-book pass + five reads + five verification passes
                tok[k_] = float((tok_all + tok_pv).mean())
        if q.c_keep.notna().all():
            acc["ev_keep"], tok["ev_keep"] = q.c_keep.to_numpy().astype(float), float((q.tok_ne + q.tok_keep).mean())
        grp = q.group.to_numpy()
        boot = lambda f: gate.cluster_bootstrap(grp, f, n_boot=a.n_boot, alpha=a.alpha)
        sel = picks["pick_benefit"]
        rep[d] = dict(n=len(q), have_pv=have_pv, ne_rerun_agreement=float(np.nanmean(q.ne_agree)) if q.ne_agree.notna().any() else None,
                      acc={k_: boot(lambda i, v=v: float(v[i].mean())) for k_, v in acc.items()},
                      minus_always={k_: boot(lambda i, v=v: float(v[i].mean() - acc["always"][i].mean())) for k_, v in acc.items() if k_ != "always"},
                      minus_never={k_: boot(lambda i, v=v: float(v[i].mean() - acc["never"][i].mean())) for k_, v in acc.items() if k_ != "never"},
                      tokens=tok, pick_rate={int(k): float((sel == k).mean()) for k in range(K + 1)},
                      per_passage_acc={int(k): float(c[c.k == k].c.mean()) for k in range(1, K + 1)},
                      benefit_minus_correct=boot(lambda i: float(acc["pick_benefit"][i].mean() - acc["pick_correct"][i].mean())))
    go = None
    if a.split == "dev" and not a.train_run and all(d in rep for d in ("evqa", "infoseek", "okvqa")):
        gains = [rep[d]["minus_always"]["pick_benefit"] for d in ("evqa", "infoseek")]  # (point, lo, hi)
        ok_vs_never = rep["okvqa"]["minus_never"]["pick_benefit"]
        go = bool(all(v[0] >= 0.03 and v[1] > 0 for v in gains) and ok_vs_never[0] >= -0.01)
        rep["go"] = go
    out = pathlib.Path(a.run) / f"passage_select__{a.model}__{a.split}"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1, default=float))
    pc = int(round(100 * (1 - a.alpha)))
    f = lambda v: f"{100*v[0]:.1f} [{100*v[1]:.1f}, {100*v[2]:.1f}]"
    g_ = lambda v: f"{100*v[0]:+.1f} [{100*v[1]:+.1f}, {100*v[2]:+.1f}]"
    L = [f"# Passage-level selection (PREREG v3): {a.model}, split {a.split}, {'fit on dev of ' + a.train_run if a.train_run else 'grouped 5-fold out of fold'}; {pc}% intervals\n"]
    for d in logged:
        r = rep[d]
        agree = r["ne_rerun_agreement"]
        agree_s = "n/a" if agree is None or agree != agree else f"{100 * agree:.1f}%"
        L += [f"## {d} (n={r['n']}; verification pass {'available' if r['have_pv'] else 'missing'}; closed-book re-run agreement {agree_s})\n",
              "| policy | accuracy | − always | − never | tokens / query |", "|---|---|---|---|---|"]
        for k_ in r["acc"]:
            L.append(f"| {k_} | {f(r['acc'][k_])} | {g_(r['minus_always'][k_]) if k_ in r['minus_always'] else '—'} | "
                     f"{g_(r['minus_never'][k_]) if k_ in r['minus_never'] else '—'} | {r['tokens'].get(k_, float('nan')):.0f} |")
        L.append(f"\nVGate-pick chooses: " + ", ".join(f"{'closed-book' if k == 0 else 'passage ' + str(k)} {100*v:.0f}%" for k, v in r["pick_rate"].items())
                 + f". Accuracy reading passage k alone: " + ", ".join(f"k={k} {100*v:.1f}" for k, v in r["per_passage_acc"].items())
                 + f". Benefit target − correctness target: {g_(r['benefit_minus_correct'])}.\n")
    if go is not None:
        L.append(f"**Go / no-go (PREREG v3): {'GO' if go else 'NO-GO'}** — VGate-pick − always ≥ +3.0 with the interval above 0 on E-VQA and InfoSeek, "
                 f"and not more than 1.0 below never-retrieve on OK-VQA.")
    pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

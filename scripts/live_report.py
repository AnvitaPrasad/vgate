"""Report of the live-pipeline system check (PREREG v2f) from the J9 outputs.

  python scripts/live_report.py [--run experiments/runs/J9] [--model Qwen3-VL-4B-Instruct]
Writes experiments/results/live_check.{json,md}:
  1. equivalence with the logged runs: closed-book answers, u_pre (live) vs frozen-gate u_pre on logged hidden states;
  2. OK-VQA: live top-5 vs logged top-5 (overlap, answer-string pseudo recall@5);
  3. accuracy of never / always (live retriever) / VGate (budget-free rule), search and read rates, 95% cluster bootstrap;
  4. per-stage wall time and tokens.
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from vgate import gate, uncertainty as U  # noqa: E402
from vgate.evaluation import (BEMScorer, evqa_correct_batch, infoseek_correct, load_infoseek_refs, okvqa_correct,  # noqa: E402
                              preprocess_answer)

LOGGED = dict(evqa="J4", infoseek="J4", okvqa="J6")  # held-out runs of the main model


def correct(d, preds, recs, scorer, refs):
    if d == "evqa":
        return evqa_correct_batch(preds, [r["question"] for r in recs], [r["answers"] for r in recs], scorer=scorer)[0].astype(int)
    if d == "okvqa":
        return np.array([okvqa_correct(p, r["answers"])[0] for p, r in zip(preds, recs)])
    return np.array([int(infoseek_correct(p, refs[r["qid"]])) for p, r in zip(preds, recs)])


def pseudo_recall(texts, answers):
    txt = " ".join(texts).lower()
    return float(any(a and re.search(r"\b" + re.escape(a) + r"\b", txt) for a in {preprocess_answer(x) for x in answers}))


def refit_oof(parts, n_boot, model):
    """PREREG v2g: refit VGate-pre / VGate-full on the live system's own logs, scored out of fold (5 folds, grouped by
    image, stratified by dataset). parts: [(dataset, rows, recs, c0, cE, hidden, logged_ne)]."""
    import operating_point as OP
    from sklearn.model_selection import StratifiedGroupKFold
    X, H, c0, cE, grp, ds, s_tr, r_tr = [], [], [], [], [], [], [], []
    for d, rows, recs, a0, aE, hid, ne in parts:
        for r, rec, h in zip(rows, recs, hid):
            feats = {**U.free_features(ne[r["qid"]]), **U.qpp(r["retr_scores"])}  # token statistics from the logged NE run
            X.append([feats[k] for k in OP.BASE]); H.append(h.astype(np.float32))
            grp.append(f"{d}/{rec['image_id']}"); ds.append(d); s_tr.append(r["search"]); r_tr.append(r["read"])
        c0 += list(a0); cE += list(aE)
    X, H, c0, cE, grp, ds = np.array(X), np.array(H), np.array(c0, float), np.array(cE, float), np.array(grp), np.array(ds)
    u, pre, full = (cE - c0).astype(int), np.zeros(len(c0)), np.zeros(len(c0))
    # raw gate inputs and outcomes of the live queries, for the one-sided-feedback simulation (scripts/online_gate.py)
    np.savez_compressed(ROOT / "experiments/results/live_stream.npz", base=X, hidden=H, c0=c0, c1=cE, dataset=ds)
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, ds, grp):
        G = OP.fit_benefit_gates(pd.DataFrame(X[tr], columns=OP.BASE).assign(u=u[tr]), H[tr])
        P = G["pca"].transform(G["sc"].transform(H[te]))
        pre[te], full[te] = OP.benefit_score(G["pre"], P), OP.benefit_score(G["full"], np.hstack([np.nan_to_num(X[te]), P]))
    # the gates a live system should run: the same fit on all logged live queries, frozen next to the transferred ones
    import freeze_gates as FG
    from vgate.live import GATES_DIR
    G_all = OP.fit_benefit_gates(pd.DataFrame(X, columns=OP.BASE).assign(u=u), H)
    meta0 = json.loads((GATES_DIR / f"{model}.json").read_text())
    g_live = FG.export(G_all, meta0["layer"], len(u), GATES_DIR / f"{model}__live.npz")
    P_all = G_all["pca"].transform(G_all["sc"].transform(H))
    d_pre = float(np.abs(OP.benefit_score(G_all["pre"], P_all) - g_live.pre(H)).max())
    d_full = float(np.abs(OP.benefit_score(G_all["full"], np.hstack([np.nan_to_num(X), P_all])) - g_live.full(X, H)).max())
    assert d_pre < 1e-4 and d_full < 1e-4, (d_pre, d_full)
    (GATES_DIR / f"{model}__live.json").write_text(json.dumps(dict(
        model=model, layer=meta0["layer"], n_train=int(len(u)), base=OP.BASE, fitted_on="all J9 live logs (PREREG v2g); out-of-fold "
        "accuracy of this procedure is in experiments/results/live_check.json", retriever="BAAI/bge-base-en-v1.5 over the M2KR OK-VQA passages, "
        "query = VLM caption + question", max_abs_diff_vs_sklearn=dict(pre=d_pre, full=d_full)), indent=1))
    pol = dict(never=np.zeros(len(c0), bool), always=np.ones(len(c0), bool), transferred=np.array(r_tr, bool),
               refit=(pre > 0) & (full > 0), refit_pre_only=pre > 0)
    srch = dict(never=pol["never"], always=pol["always"], transferred=np.array(s_tr, bool), refit=pre > 0, refit_pre_only=pre > 0)
    acc = {k: np.where(v, cE, c0) for k, v in pol.items()}
    names = list(dict.fromkeys(ds))
    macro = lambda x, i: float(np.mean([x[i][ds[i] == d].mean() for d in names]))
    strata = np.array([names.index(d) for d in ds])
    boot = lambda a_, b_: gate.cluster_bootstrap(grp, lambda i: macro(acc[a_], i) - macro(acc[b_], i), n_boot=n_boot, alpha=0.05, strata=strata)
    allidx = np.arange(len(c0))
    return dict(n=len(c0), help_rate={d: float((u[ds == d] == 1).mean()) for d in names}, harm_rate={d: float((u[ds == d] == -1).mean()) for d in names},
                policies={k: dict(macro_acc=macro(acc[k], allidx), **{d: dict(acc=float(acc[k][ds == d].mean()), read=float(pol[k][ds == d].mean()),
                                                                             search=float(srch[k][ds == d].mean())) for d in names}) for k in pol},
                macro_diff_95ci={f"{a_} - {b_}": boot(a_, b_) for a_, b_ in [("refit", "transferred"), ("refit", "never"), ("refit", "always"),
                                                                              ("transferred", "never"), ("refit_pre_only", "never")]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "experiments/runs/J9"))
    ap.add_argument("--model", default="Qwen3-VL-4B-Instruct")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    frozen = pd.read_csv(ROOT / f"experiments/results/frozen_gate_scores__{a.model}.csv", index_col="qid")
    frozen.index = frozen.index.astype(str)
    scorer, refs, rep, parts = None, None, {}, []
    for d in ("okvqa", "evqa", "infoseek"):
        p = run / f"live__{d}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p)]
        allrec = {r["qid"]: r for r in map(json.loads, open(ROOT / f"data/processed/{d}.jsonl"))}
        recs = [allrec[r["qid"]] for r in rows]
        if d == "evqa" and scorer is None:
            scorer = BEMScorer()
        if d == "infoseek" and refs is None:
            refs = load_infoseek_refs()
        c0 = correct(d, [r["a0"] for r in rows], recs, scorer, refs)
        cE = correct(d, [r["aE"] for r in rows], recs, scorer, refs)
        read, search = np.array([r["read"] for r in rows]), np.array([r["search"] for r in rows])
        cV = np.where(read, cE, c0)
        grp = np.array([str(r["image_id"]) for r in recs])
        boot = lambda f: gate.cluster_bootstrap(grp, f, n_boot=a.n_boot, alpha=0.05)
        logged_ne = U.read(ROOT / "experiments/runs" / LOGGED[d], d, a.model, "ne")
        same_a0 = np.array([U.norm(r["a0"]) == U.norm(logged_ne[r["qid"]]["text"]) for r in rows])
        u_live, u_frz = np.array([r["u_pre"] for r in rows]), frozen.u_pre.reindex([r["qid"] for r in rows]).to_numpy()
        sec = lambda k: np.array([r["seconds"].get(k, np.nan) for r in rows])
        out = dict(n=len(rows),
                   closed_book_answer_agreement=float(same_a0.mean()),
                   u_pre_corr=float(np.corrcoef(u_live, u_frz)[0, 1]), u_pre_sign_agreement=float(((u_live > 0) == (u_frz > 0)).mean()),
                   u_pre_max_abs_diff=float(np.abs(u_live - u_frz).max()),
                   acc_never=boot(lambda i: float(c0[i].mean())), acc_always_live_retriever=boot(lambda i: float(cE[i].mean())),
                   acc_vgate=boot(lambda i: float(cV[i].mean())),
                   vgate_minus_always=boot(lambda i: float(cV[i].mean() - cE[i].mean())),
                   vgate_minus_never=boot(lambda i: float(cV[i].mean() - c0[i].mean())),
                   logged_acc_never=float(frozen.c_ne.reindex([r["qid"] for r in rows]).mean()),
                   logged_acc_always_benchmark_retriever=float(frozen.c_ev.reindex([r["qid"] for r in rows]).mean()),
                   search_rate=float(search.mean()), read_rate=float(read.mean()),
                   seconds={k: dict(median=float(np.nanmedian(sec(k))), p90=float(np.nanpercentile(sec(k), 90)))
                            for k in ("closed_book", "caption", "retrieval", "evidence")},
                   tokens=dict(closed_book=float(np.mean([r["tok_closed_book"] for r in rows])),
                               caption=float(np.mean([r["tok_caption"] for r in rows])),
                               evidence=float(np.mean([r["tok_evidence"] for r in rows])),
                               vgate=float(np.mean([r["tok_vgate"] for r in rows]))))
        if d == "okvqa":  # retrieval with a VLM caption (live) vs the dataset caption (logged)
            pq = pd.read_parquet(ROOT / "data/raw/m2kr/OKVQA_passages/test_passages-00000-of-00001.parquet").drop_duplicates("passage_id")
            text = dict(zip(pq.passage_id.astype(str), pq.passage_content))
            out["top5_overlap_with_logged"] = float(np.mean([len({e["id"] for e in r["evidence"]} & {e["id"] for e in rec["evidence"]}) / 5
                                                              for r, rec in zip(rows, recs)]))
            out["pseudo_recall5_live"] = float(np.mean([pseudo_recall([text[e["id"]] for e in r["evidence"]], rec["answers"])
                                                        for r, rec in zip(rows, recs)]))
            out["pseudo_recall5_logged"] = float(np.mean([pseudo_recall([e["title"] + " " + e["text"] for e in rec["evidence"]], rec["answers"])
                                                          for rec in recs]))
        rep[d] = out
        parts.append((d, rows, recs, c0, cE, np.load(run / f"live_hidden__{d}.npy"), logged_ne))
    rep["refit_on_live_logs"] = refit_oof(parts, a.n_boot, a.model)
    res = ROOT / "experiments/results/live_check"
    pathlib.Path(str(res) + ".json").write_text(json.dumps(rep, indent=1, default=float))
    f = lambda v: f"{100*v[0]:.1f} [{100*v[1]:.1f}, {100*v[2]:.1f}]"
    g = lambda v: f"{100*v[0]:+.1f} [{100*v[1]:+.1f}, {100*v[2]:+.1f}]"
    L = ["# Live-pipeline system check (PREREG v2f)", f"Run: `{run.relative_to(ROOT)}`, model {a.model}; every pass executed for every query, one query at a time.\n",
         "| dataset | n | closed-book answers = logged | u_pre corr. | u_pre sign agreement | search | read |", "|---|---|---|---|---|---|---|"]
    refit = rep.pop("refit_on_live_logs")
    L += [f"| {d} | {r['n']} | {100*r['closed_book_answer_agreement']:.1f}% | {r['u_pre_corr']:.4f} | {100*r['u_pre_sign_agreement']:.1f}% | "
          f"{100*r['search_rate']:.0f}% | {100*r['read_rate']:.0f}% |" for d, r in rep.items()]
    L += ["\n| dataset | never | always (live retriever) | VGate (budget-free) | VGate − always | VGate − never | logged never / always (benchmark retriever) |",
          "|---|---|---|---|---|---|---|"]
    L += [f"| {d} | {f(r['acc_never'])} | {f(r['acc_always_live_retriever'])} | {f(r['acc_vgate'])} | {g(r['vgate_minus_always'])} | "
          f"{g(r['vgate_minus_never'])} | {100*r['logged_acc_never']:.1f} / {100*r['logged_acc_always_benchmark_retriever']:.1f} |" for d, r in rep.items()]
    L += ["\n| dataset | closed-book s (median / p90) | caption s | retrieval s | evidence s | tokens: closed-book / caption / evidence / VGate |",
          "|---|---|---|---|---|---|"]
    L += [f"| {d} | " + " | ".join(f"{r['seconds'][k]['median']:.2f} / {r['seconds'][k]['p90']:.2f}" for k in ("closed_book", "caption", "retrieval", "evidence"))
          + f" | {r['tokens']['closed_book']:.0f} / {r['tokens']['caption']:.0f} / {r['tokens']['evidence']:.0f} / {r['tokens']['vgate']:.0f} |"
          for d, r in rep.items()]
    if "okvqa" in rep and "top5_overlap_with_logged" in rep["okvqa"]:
        o = rep["okvqa"]
        L.append(f"\nOK-VQA retrieval with a VLM caption vs the dataset caption: top-5 overlap {100*o['top5_overlap_with_logged']:.0f}%; "
                 f"answer-string pseudo recall@5 {100*o['pseudo_recall5_live']:.0f}% (live) vs {100*o['pseudo_recall5_logged']:.0f}% (logged).")
    names = [d for d in rep]
    L += ["\n## Gates refit on the live system's own logs (PREREG v2g; out of fold, exploratory)\n",
          "Retrieval with the live retriever helps / harms: " + ", ".join(f"{d} {100*refit['help_rate'][d]:.1f}% / {100*refit['harm_rate'][d]:.1f}%" for d in names) + ".\n",
          "| policy | " + " | ".join(f"{d} acc (search / read)" for d in names) + " | macro |", "|---|" + "---|" * (len(names) + 1)]
    for k, v in refit["policies"].items():
        L.append(f"| {k} | " + " | ".join(f"{100*v[d]['acc']:.1f} ({100*v[d]['search']:.0f}% / {100*v[d]['read']:.0f}%)" for d in names) + f" | {100*v['macro_acc']:.1f} |")
    L += ["\n| macro accuracy difference | point | 95% lo | 95% hi |", "|---|---|---|---|"]
    L += [f"| {k} | {100*v[0]:+.2f} | {100*v[1]:+.2f} | {100*v[2]:+.2f} |" for k, v in refit["macro_diff_95ci"].items()]
    rep["refit_on_live_logs"] = refit
    pathlib.Path(str(res) + ".json").write_text(json.dumps(rep, indent=1, default=float))
    pathlib.Path(str(res) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

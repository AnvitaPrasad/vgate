"""Pilot of passage selection with a released, trained critic (PREREG v5, analysis 2; development data; written before
the critic outputs existed). The critic's scores sit in the run folder as the `pv_k` files of the reader model, so that
scripts/passage_select.py (analysis 1) runs unchanged.

  python scripts/critic_pilot.py --run experiments/runs/J13 --logged evqa=J1,infoseek=J1,okvqa=J6
      -> experiments/results/critic_pilot.{json,md}
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import loss_decomposition as LD  # noqa: E402
import passage_select as PS  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

K, THR = PS.K, 0.1  # the critic's card keeps a passage when the raw probability of "Yes" exceeds 0.1


def load(run, logged, d, model, split):
    c, q, _ = PS.candidates(run, logged, ROOT / "data", d, model, split)
    raw = {k: U.read(run, d, model, f"pv_{k}") for k in range(1, K + 1)}
    c["raw"] = [raw[int(k)][i].get("yes_raw", raw[int(k)][i]["p_yes"]) for i, k in zip(c.qid, c.k)]  # the reader's own check stores p_yes only
    return c, q


def critic_pick(c, q):
    """Passage with the highest critic score if its raw P(Yes) exceeds THR, else 0 (closed-book). No fitted parameter."""
    b = c.sort_values(["qid", "raw", "k"], ascending=[True, False, True]).groupby("qid").head(1).set_index("qid").reindex(q.index)
    return pd.Series(np.where(b.raw > THR, b.k, 0).astype(int), index=q.index)


def acc_of(sel, c, q):
    g = c.set_index(["qid", "k"]).c
    return np.array([q.c_ne[i] if k == 0 else g[(i, int(k))] for i, k in sel.items()], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--logged", required=True)
    ap.add_argument("--model", default="Qwen3-VL-4B-Instruct")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--tag", default="", help="suffix of the output files (e.g. reader_check for the reader's own yes/no check, J10)")
    a = ap.parse_args()
    R, rep, parts = ROOT / "experiments/runs", {}, {}
    f3 = lambda v: [round(100 * x, 2) for x in v]
    for d, lr in dict(x.split("=") for x in a.logged.split(",")).items():
        c, q = load(a.run, R / lr, d, a.model, a.split)
        sel = critic_pick(c, q); acc = acc_of(sel, c, q); alw, nev = q.c_ev.to_numpy().astype(float), q.c_ne.to_numpy().astype(float)
        boot = lambda f, g=q.group.to_numpy(): f3(gate.cluster_bootstrap(g, f, n_boot=a.n_boot, alpha=0.2))
        o = dict(n=len(q), always=round(100 * alw.mean(), 2), never=round(100 * nev.mean(), 2), critic_pick=round(100 * acc.mean(), 2),
                 minus_always=boot(lambda i: float(acc[i].mean() - alw[i].mean())), minus_never=boot(lambda i: float(acc[i].mean() - nev[i].mean())),
                 answers_closed_book=round(100 * float((sel == 0).mean()), 1), accepted_per_question=round(float((c.raw > THR).groupby(c.qid).sum().mean()), 2),
                 tokens=round(float((q.tok_ne + c.groupby("qid").tok_pv.sum().reindex(q.index)
                                     + np.array([0 if k == 0 else c[(c.qid == i) & (c.k == k)].tok.iloc[0] for i, k in sel.items()])).mean()), 1),
                 tokens_always=round(float(q.tok_ev.mean()), 1))
        if d in ("evqa", "infoseek"):  # gold labels: analysis only
            G = LD.gold_info(d); c["gold_rank"] = c.qid.map(G["rank"]); c["is_gold"] = (c.k == c.gold_rank).astype(int)
            g = c[c.gold_rank > 0]
            top = g.sort_values(["qid", "raw", "k"], ascending=[True, False, True]).groupby("qid").head(1)
            o.update(gold_in_top5=round(100 * float((c.groupby("qid").gold_rank.first() > 0).mean()), 1),
                     top1_is_gold=round(100 * float((g.groupby("qid").gold_rank.first() == 1).mean()), 1),  # added after the run: the retriever's first rank
                     argmax_is_gold=round(100 * float(top.is_gold.mean()), 1), auroc_gold=round(float(roc_auc_score(g.is_gold, g.raw)), 3),
                     recall_at_thr=round(100 * float((g[g.is_gold == 1].raw > THR).mean()), 1),
                     specificity_at_thr=round(100 * float((c[c.is_gold == 0].raw <= THR).mean()), 1),
                     gold_alone_acc=round(100 * float(c[c.is_gold == 1].c.mean()), 1))
        rep[d] = o; parts[d] = (c, q, acc)
    # benchmark-identity control: macro over the benchmarks against the better fixed policy of each
    names = list(parts)
    grp = np.concatenate([parts[d][1].group.to_numpy() for d in names]); strata = np.concatenate([np.full(len(parts[d][1]), j) for j, d in enumerate(names)])
    offs = np.cumsum([0] + [len(parts[d][1]) for d in names])
    fixed = [parts[d][1].c_ev.to_numpy().astype(float) if parts[d][1].c_ev.mean() >= parts[d][1].c_ne.mean() else parts[d][1].c_ne.to_numpy().astype(float) for d in names]

    def macro_diff(accs):
        def stat(idx):
            return float(np.mean([(accs[j][i] - fixed[j][i]).mean() for j, (lo, hi) in enumerate(zip(offs[:-1], offs[1:])) for i in [idx[(idx >= lo) & (idx < hi)] - lo]]))
        return f3(gate.cluster_bootstrap(grp, stat, n_boot=a.n_boot, alpha=0.2, strata=strata))
    pick_accs = [parts[d][2] for d in names]
    # one benefit-trained selector for all traffic, no dataset identity (out of fold over the pooled dev sets)
    C = pd.concat([parts[d][0] for d in names], ignore_index=True); Q = pd.concat([parts[d][1] for d in names])
    qs, res = Q.index.to_numpy(), {}
    for tr, te in GroupKFold(n_splits=5).split(qs, groups=Q.group.to_numpy()):
        for k_, v in PS.select(C, Q, set(qs[tr]), set(qs[te]), PS.FEATS).items():
            res[k_] = pd.concat([res[k_], v]) if k_ in res else v
    pooled = res["pick_benefit"].reindex(Q.index)
    pooled_accs = [acc_of(pooled.loc[parts[d][1].index], parts[d][0], parts[d][1]) for d in names]
    rep["macro"] = dict(fixed_per_benchmark=round(100 * float(np.mean([f.mean() for f in fixed])), 2),
                        critic_pick=round(100 * float(np.mean([x.mean() for x in pick_accs])), 2), critic_pick_minus_fixed=macro_diff(pick_accs),
                        pooled_selector=round(100 * float(np.mean([x.mean() for x in pooled_accs])), 2), pooled_selector_minus_fixed=macro_diff(pooled_accs),
                        pooled_selector_by_benchmark={d: round(100 * float(x.mean()), 2) for d, x in zip(names, pooled_accs)})
    exp = dict(P1=bool(rep.get("evqa", {}).get("argmax_is_gold", 0) >= 80), P2=bool(rep["evqa"]["minus_always"][0] >= 3.0 and rep["evqa"]["minus_always"][1] > 0),
               P3=bool(rep["macro"]["critic_pick_minus_fixed"][0] >= 2.0))
    f = lambda v: f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"
    L = [f"# Passage selection with the released critic (PREREG v5 pilot, {a.split} split, 80% intervals)", "",
         "| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for key, name in (("never", "never"), ("always", "always (top-5)"), ("critic_pick", "critic-pick (no fitted parameter)"), ("answers_closed_book", "… answers closed-book (%)"),
                      ("accepted_per_question", "passages accepted per question"), ("tokens", "tokens per query"), ("tokens_always", "tokens, always"),
                      ("gold_in_top5", "gold in top-5 (%)"), ("argmax_is_gold", "arg-max critic is the gold passage (%)"), ("auroc_gold", "AUROC for the gold passage"),
                      ("recall_at_thr", "recall at the card's threshold"), ("specificity_at_thr", "specificity at the card's threshold")):
        L.append(f"| {name} | " + " | ".join(str(rep[d].get(key, "—")) for d in names) + " |")
    L += ["| critic-pick − always | " + " | ".join(f(rep[d]["minus_always"]) for d in names) + " |",
          "| critic-pick − never | " + " | ".join(f(rep[d]["minus_never"]) for d in names) + " |", "",
          f"Macro over the benchmarks: better fixed policy per benchmark {rep['macro']['fixed_per_benchmark']}; critic-pick {rep['macro']['critic_pick']} "
          f"({f(rep['macro']['critic_pick_minus_fixed'])}); one benefit-trained selector without dataset identity {rep['macro']['pooled_selector']} "
          f"({f(rep['macro']['pooled_selector_minus_fixed'])}; {rep['macro']['pooled_selector_by_benchmark']}).", "",
          "## Expectations (PREREG v5)", ""] + [f"- {k}: {'held' if v else 'NOT held'}" for k, v in exp.items()]
    out = ROOT / "experiments/results" / ("critic_pilot" + (f"__{a.tag}" if a.tag else ""))
    out.with_suffix(".json").write_text(json.dumps(dict(results=rep, expectations=exp, threshold=THR), indent=1, default=float))
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

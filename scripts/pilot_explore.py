"""Exploratory analyses of the passage-selection pilot (PREREG v3 addendum E1-E3; development data; written before the
pilot outputs existed). None of this enters the go / no-go rule.

  python scripts/pilot_explore.py --run experiments/runs/J10 --logged evqa=J1,infoseek=J1,okvqa=J6
      -> experiments/results/pilot_explore.{json,md}
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import loss_decomposition as LD  # noqa: E402
import passage_select as PS  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

K = PS.K
QF = U.FREE + ["qpp_top1_z", "qpp_margin_z", "qpp_nqc", "qpp_entropy"]  # closed-book and retriever-score features only


def e1(c, q, G):
    """Share of questions whose top-5 answer equals the answer from passage k alone, by the rank of the gold passage."""
    a5 = q.ev_text.map(U.norm)
    same = c.assign(same=c.text.map(U.norm).to_numpy() == c.qid.map(a5).to_numpy()).pivot(index="qid", columns="k", values="same").reindex(q.index)
    out = {"all": {int(k): round(100 * float(same[k].mean()), 1) for k in range(1, K + 1)},
           "top5_equals_closed_book": round(100 * float((a5 == q.ne_text.map(U.norm)).mean()), 1)}
    if G is not None:
        r = G["rank"].reindex(q.index)
        for name, m in (("gold_rank_1", r == 1), ("gold_rank_2to5", r > 1), ("gold_absent", r == 0)):
            out[name] = {"n": int(m.sum()), **{int(k): round(100 * float(same[k][m].mean()), 1) for k in range(1, K + 1)}}
        m = r > 1  # gold at ranks 2-5: does the answer follow passage 1 or the gold passage?
        gold_same = np.array([same.at[i, int(r[i])] for i in q.index[m]])
        out["gold_rank_2to5_follows"] = dict(passage_1=round(100 * float(same[1][m].mean()), 1), gold_passage=round(100 * float(gold_same.mean()), 1))
    return out


def e2(c, q, boot):
    """Three-way reading-budget choice {closed-book, top-1 passage alone, top-5} from question-level features, out of fold."""
    c1 = c[c.k == 1].set_index("qid").reindex(q.index)
    X, grp = np.nan_to_num(q[QF].to_numpy()), q.group.to_numpy()
    u1, u5 = (c1.c - q.c_ne).to_numpy().astype(int), (q.c_ev - q.c_ne).to_numpy().astype(int)
    s1, s5 = np.zeros(len(q)), np.zeros(len(q))
    for tr, te in GroupKFold(n_splits=5).split(X, groups=grp):
        s1[te] = PS.benefit_score(PS.fit_benefit(X[tr], u1[tr]), X[te])
        s5[te] = PS.benefit_score(PS.fit_benefit(X[tr], u5[tr]), X[te])
    pick = np.where((s1 <= 0) & (s5 <= 0), 0, np.where(s1 >= s5, 1, 5))
    C = np.c_[q.c_ne, c1.c, q.c_ev].astype(float)
    # tokens as in the paper's ledger: always-retrieve pays the evidence pass only; a policy that chooses pays the closed-book
    # pass plus what it reads (corrected after the first run, which charged always-retrieve the closed-book pass as well)
    T = np.c_[q.tok_ne, q.tok_ne + c1.tok, q.tok_ev].astype(float)
    T_gate = np.c_[q.tok_ne, q.tok_ne + c1.tok, q.tok_ne + q.tok_ev].astype(float)
    col = np.select([pick == 0, pick == 1], [0, 1], 2)
    acc = dict(never=C[:, 0], top1_alone=C[:, 1], always=C[:, 2], allocator=C[np.arange(len(q)), col], oracle3=C.max(1))
    tok = dict(never=T[:, 0].mean(), top1_alone=T[:, 1].mean(), always=T[:, 2].mean(), allocator=T_gate[np.arange(len(q)), col].mean())
    return dict(acc={k: boot(lambda i, v=v: float(v[i].mean())) for k, v in acc.items()},
                allocator_minus_always=boot(lambda i: float(acc["allocator"][i].mean() - acc["always"][i].mean())),
                top1_minus_always=boot(lambda i: float(acc["top1_alone"][i].mean() - acc["always"][i].mean())),
                tokens={k: round(float(v), 1) for k, v in tok.items()}, share={int(k): round(float((pick == k).mean()), 3) for k in (0, 1, 5)})


def e3(c, q, boot, have_pv):
    """Early exit: read passages in retriever order, stop at the first with a positive predicted benefit (out of fold)."""
    feats = [f for f in PS.FEATS if f not in PS.PVF and f != "n_agree"]  # only what is known when passage k has just been read
    qs, s = q.index.to_numpy(), pd.Series(0.0, index=c.index)
    for tr, te in GroupKFold(n_splits=5).split(qs, groups=q.group.to_numpy()):
        a, b = c.qid.isin(set(qs[tr])), c.qid.isin(set(qs[te]))
        s[b] = PS.benefit_score(PS.fit_benefit(c[a][feats].to_numpy(), c[a].u.to_numpy()), c[b][feats].to_numpy())
    S = c.assign(s=s).pivot(index="qid", columns="k", values="s").reindex(q.index).to_numpy()
    Ck = c.pivot(index="qid", columns="k", values="c").reindex(q.index).to_numpy()
    Tk = c.pivot(index="qid", columns="k", values="tok").reindex(q.index).to_numpy()
    out = {}
    for cap in (2, 3, 5):
        hit = S[:, :cap] > 0
        stop = np.where(hit.any(1), hit.argmax(1), -1)            # index of the passage whose answer is taken, -1 = closed-book
        n_read = np.where(stop >= 0, stop + 1, cap)
        acc = np.where(stop >= 0, Ck[np.arange(len(q)), np.maximum(stop, 0)], q.c_ne.to_numpy()).astype(float)
        tok = q.tok_ne.to_numpy() + np.array([Tk[i, :n].sum() for i, n in enumerate(n_read)])
        out[f"cap{cap}"] = dict(acc=boot(lambda i, v=acc: float(v[i].mean())),
                                minus_always=boot(lambda i, v=acc: float(v[i].mean() - q.c_ev.to_numpy()[i].mean())),
                                tokens=round(float(tok.mean()), 1), passages_read=round(float(n_read.mean()), 2),
                                answered_closed_book=round(float((stop < 0).mean()), 3))
    out["always_tokens"] = round(float(q.tok_ev.mean()), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--logged", required=True)
    ap.add_argument("--model", default="Qwen3-VL-4B-Instruct")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    R, data, rep = ROOT / "experiments/runs", ROOT / "data", {}
    for d, lr in dict(x.split("=") for x in a.logged.split(",")).items():
        c, q, have_pv = PS.candidates(a.run, R / lr, data, d, a.model, a.split)
        boot = lambda f, g=q.group.to_numpy(): [round(100 * v, 2) for v in gate.cluster_bootstrap(g, f, n_boot=a.n_boot, alpha=0.2)]
        rep[d] = dict(n=len(q), E1=e1(c, q, LD.gold_info(d) if d in ("evqa", "infoseek") else None), E2=e2(c, q, boot), E3=e3(c, q, boot, have_pv))
    f = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"
    lines = [f"# Pilot, exploratory analyses E1–E3 ({a.split} split, 80% intervals; PREREG v3 addendum)", ""]
    for d, r in rep.items():
        lines += [f"## {d} (n={r['n']})", "", "**E1** top-5 answer equals the answer from passage k alone (%): "
                  + ", ".join(f"k={k}: {v}" for k, v in r["E1"]["all"].items()) + f"; equals the closed-book answer: {r['E1']['top5_equals_closed_book']}."]
        if "gold_rank_2to5_follows" in r["E1"]:
            g = r["E1"]["gold_rank_2to5_follows"]
            lines.append(f"With the gold passage at ranks 2–5 (n={r['E1']['gold_rank_2to5']['n']}): the top-5 answer equals passage 1's answer in "
                         f"{g['passage_1']}% and the gold passage's answer in {g['gold_passage']}%.")
        e = r["E2"]
        lines += ["", "**E2** reading-budget choice: " + "; ".join(f"{k} {f(v)} ({e['tokens'].get(k, '—')} tok)" for k, v in e["acc"].items())
                  + f". Allocator − always {f(e['allocator_minus_always'])}; top-1 alone − always {f(e['top1_minus_always'])}; shares {e['share']}.", "",
                  "**E3** early exit: " + "; ".join(f"cap {k[3:]}: {f(v['acc'])}, − always {f(v['minus_always'])}, {v['tokens']} tok, {v['passages_read']} passages"
                                                     for k, v in r["E3"].items() if k.startswith("cap")) + f" (always: {r['E3']['always_tokens']} tok).", ""]
    out = ROOT / "experiments/results/pilot_explore"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=float))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

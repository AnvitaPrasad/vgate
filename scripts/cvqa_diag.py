"""EXPLORATORY diagnosis of the CVQA result (PREREG v7: G1 not held): why does the gate not lead the better fixed policy
by a point when an oracle gate has 8 points of headroom, and which fix has a chance? Nothing here is a registered result.
Parts 1-4 describe the held-out outcome; part 5 tries candidate rules on DEVELOPMENT data only.

  python scripts/cvqa_diag.py   -> experiments/results/cvqa_diag.md
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import cvqa_audit as CA  # noqa: E402
import operating_point as OP  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

R = ROOT / "experiments/runs"
DATA = R / "J14/data"
L = []


def say(s=""):
    L.append(s); print(s)


def acc_of(read, fr):
    return float(np.where(read, fr.c_ev, fr.c_ne).mean())


tr, Htr, seg_tr = CA.load(R / "J14", DATA, "dev")
te, Hte, seg_te = CA.load(R / "J15", DATA, "confirm")
D, S = CA.decide(tr, Htr, te, Hte, seg_tr, seg_te)
u, c0, c1 = te.u.to_numpy(), te.c_ne.to_numpy().astype(float), te.c_ev.to_numpy().astype(float)
never = c0.mean()
say("# CVQA: why the gate does not lead the better fixed policy (exploratory; see PREREG v7 for the registered outcome)")
say()
say("## 1. What the gate reads (held-out, gate fitted on development data)")
read = D["benefit_full"]
for name, m in (("read", read), ("skipped", ~read)):
    say(f"- {name}: {100 * m.mean():.1f}% of questions; of these retrieval helps {100 * (u[m] == 1).mean():.1f}%, harms {100 * (u[m] == -1).mean():.1f}%, "
        f"changes nothing {100 * (u[m] == 0).mean():.1f}%")
say(f"- net effect of its reads: {100 * (c1[read] - c0[read]).sum() / len(te):+.2f} points (it must find help without harm: of all help cases it reads "
    f"{100 * read[u == 1].mean():.1f}%, of all harm cases {100 * read[u == -1].mean():.1f}%)")
say()
say("## 2. How well the score ranks queries (held-out)")
sc = S["benefit_full"]
say(f"- AUROC of the predicted benefit: help vs harm {gate.auroc(sc[u != 0], u[u != 0] > 0):.3f}; help vs everything else {gate.auroc(sc, u > 0):.3f}; "
    f"harm vs everything else (score negated) {gate.auroc(-sc, u < 0):.3f}")
order = np.argsort(-sc); gain = np.concatenate([[0], np.cumsum((c1 - c0)[order])]) / len(te)
k = int(np.argmax(gain))
say(f"- best possible threshold on this score, chosen with the held-out labels (optimistic): read the top {100 * k / len(te):.1f}% -> {100 * gain[k]:+.2f} points over never-retrieve "
    f"(the registered rule read {100 * read.mean():.1f}% -> {100 * (acc_of(read, te) - never):+.2f})")
for q_ in (0.02, 0.05, 0.10, 0.20):
    m = np.zeros(len(te), bool); m[order[:int(q_ * len(te))]] = True
    say(f"  - top {100 * q_:.0f}% by predicted benefit: helps {100 * (u[m] == 1).mean():.1f}%, harms {100 * (u[m] == -1).mean():.1f}% (base rates {100 * (u == 1).mean():.1f}% / {100 * (u == -1).mean():.1f}%)")
say()
say("## 3. Would more labelled traffic fix it? (learning curve; held-out gain over never-retrieve of VGate-full)")
rng = np.random.default_rng(0)
grp_tr = tr.group.to_numpy(); ug = np.unique(grp_tr)
for n_ in (500, 1000, 2001):
    vals = []
    for rep in range(1 if n_ == 2001 else 5):
        keep = np.isin(grp_tr, rng.permutation(ug)[:int(len(ug) * n_ / len(tr))]) if n_ < 2001 else np.ones(len(tr), bool)
        d_, _ = CA.decide(tr[keep], Htr[keep], te, Hte, seg_tr[keep], seg_te)
        vals.append(acc_of(d_["benefit_full"], te) - never)
    say(f"- fitted on {int(np.mean(n_))} development questions: {100 * np.mean(vals):+.2f}" + (f" (range {100 * min(vals):+.2f} to {100 * max(vals):+.2f} over 5 subsamples)" if len(vals) > 1 else ""))
oof = np.zeros(len(te), bool)
for i, j in GroupKFold(n_splits=5).split(Hte, groups=te.group.to_numpy()):
    d_, _ = CA.decide(te.iloc[i], Hte[i], te.iloc[j], Hte[j], seg_te.iloc[i], seg_te.iloc[j])
    oof[j] = d_["benefit_full"]
v = gate.cluster_bootstrap(te.group.to_numpy(), lambda i: float(np.where(oof[i], c1[i], c0[i]).mean() - c0[i].mean()), n_boot=1000, alpha=0.05)
say(f"- fitted on about 6,700 held-out questions, scored on the rest (5-fold, grouped): {100 * v[0]:+.2f} [{100 * v[1]:+.2f}, {100 * v[2]:+.2f}], reading {100 * oof.mean():.1f}%")
both = pd.concat([tr, te]); Hb = np.vstack([Htr, Hte]); segb = pd.concat([seg_tr, seg_te])
say()
say("## 4. Which signals carry it (held-out gain over never-retrieve, gates fitted on development data)")
for k_ in ("benefit_post", "benefit_pre", "benefit_full", "error_post", "error_full"):
    say(f"- {CA.NAMES[k_]}: {100 * (acc_of(D[k_], te) - never):+.2f} (reads {100 * D[k_].mean():.1f}%)")
say()
say("## 5. Candidate rules, DEVELOPMENT data only (out of fold where fitted; 80% intervals)")
ne = U.read(R / "J14", "cvqa", CA.MODEL, "ne"); ev = U.read(R / "J14", "cvqa", CA.MODEL, "ev")
p0 = np.array([np.exp(ne[q]["tok_logprobs"][0]) for q in tr.index]); p1 = np.array([np.exp(ev[q]["tok_logprobs"][0]) for q in tr.index])
d0, d1, g_ = tr.c_ne.to_numpy().astype(float), tr.c_ev.to_numpy().astype(float), tr.group.to_numpy()
noletter = tr.no_letter_ev.to_numpy()


def boot(a_):
    v_ = gate.cluster_bootstrap(g_, lambda i: float(a_[i].mean() - d0[i].mean()), n_boot=1000, alpha=0.2)
    return f"{100 * v_[0]:+.2f} [{100 * v_[1]:+.2f}, {100 * v_[2]:+.2f}]"


sc_dev = pd.read_csv(R / "J14" / f"scores__cvqa__{CA.MODEL}.csv", index_col="qid").reindex(tr.index)
agree = (sc_dev.letter_ne.fillna("") == sc_dev.letter_ev.fillna("")).to_numpy()
say(f"- reference: never-retrieve {100 * d0.mean():.2f}, always-retrieve {100 * d1.mean():.2f}; the two answers name the same option on {100 * agree.mean():.1f}% of questions; "
    f"best of the two answers {100 * np.maximum(d0, d1).mean():.2f}")
say("- rules that read the evidence for every question and then choose an answer (cost: both passes), gain over never-retrieve:")
sel = {"evidence answer unless it names no option": ~noletter,
       "the answer with the higher first-token probability": (p1 > p0) & ~noletter,
       "evidence answer only if at least 0.2 more probable": (p1 - p0 > 0.2) & ~noletter}
for name, m in sel.items():
    say(f"  - {name}: {boot(np.where(m, d1, d0))} (uses the evidence answer on {100 * m.mean():.1f}%)")
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = np.column_stack([p0, p1, p1 - p0, agree, noletter, np.nan_to_num(tr[OP.BASE].to_numpy())])
uu = tr.u.to_numpy(); pick = np.zeros(len(tr), bool)
for i_, j_ in GroupKFold(n_splits=5).split(X, groups=g_):
    m_ = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(X[i_], uu[i_])
    pick[j_] = OP.benefit_score(m_, X[j_]) > 0
say(f"  - fitted selector on both passes' first-token probabilities, agreement and the gate's scalar signals (out of fold): {boot(np.where(pick, d1, d0))} "
    f"(uses the evidence answer on {100 * pick.mean():.1f}%)")
say("- the registered gate decides before reading; these rules decide after it and pay for both passes on every question.")
(ROOT / "experiments/results/cvqa_diag.md").write_text("\n".join(L) + "\n")

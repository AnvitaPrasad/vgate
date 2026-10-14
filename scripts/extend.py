"""PREREG v2b components, fit on dev and applied unchanged to the held-out split:

  VGate-full    benefit model on [free NE features, QPP, PCA-64 hidden state]            (post-search gate)
  VGate-select  after reading, return a_E or a_0: benefit model on NE + EV-pass + QPP features, threshold tau
                chosen to maximise out-of-fold DEV accuracy (H-SEL: beats always-retrieve on held-out accuracy)
  probe4        MMA-RAG's four-outcome target on the CLOSED-BOOK hidden state (a pre-search adaptation; MMA-RAG
                itself decides after both passes, see PREREG v2d), ranked by P(fixes) - P(breaks)

  python scripts/extend.py --train_run experiments/runs/J1 --test_run experiments/runs/J4 --model Qwen3-VL-4B-Instruct
Confirmatory on OK-VQA; exploratory on E-VQA / InfoSeek (see experiments/PREREG.md, v2b).
"""
import argparse
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze as A  # noqa: E402
import confirm as C  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

BASE = A.GROUPS["free"] + A.GROUPS["qpp"]
EVF = ["ev_ent_mean", "ev_ent_max", "ev_ent_first", "ev_lp_min", "ev_lp_mean", "ev_seq_lp", "ev_n_gen", "agree", "ev_abstain"]
ABSTAIN = re.compile(r"\bnot (specified|available|answerable|determinable|mentioned|provided|given|stated|known|clear)"
                     r"|\bnone\b|\bunknown\b|cannot|can't", re.I)


def add_ev_features(df, run, ds, model):
    """Token statistics of the evidence pass + agreement/abstention flags (only usable AFTER reading)."""
    ev = U.read(run, ds, model, "ev")
    rows = {}
    for q in df.index:
        e = ev[q]
        lp, en = np.array(e["tok_logprobs"] or [0.0]), np.array(e["tok_entropy"] or [0.0])
        rows[q] = dict(ev_ent_mean=en.mean(), ev_ent_max=en.max(), ev_ent_first=en[0], ev_lp_min=lp.min(),
                       ev_lp_mean=lp.mean(), ev_seq_lp=lp.sum(), ev_n_gen=e["n_gen"],
                       agree=float(U.norm(e["text"]) == U.norm(df.ne_text[q])),
                       ev_abstain=float(bool(ABSTAIN.search(e["text"] or "")) or not (e["text"] or "").strip()))
    return df.join(pd.DataFrame.from_dict(rows, orient="index"))


def benefit_model(Xtr, utr):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    return m.fit(np.nan_to_num(Xtr), utr)


def benefit_score(m, X):
    p, cls = m.predict_proba(np.nan_to_num(X)), list(m[-1].classes_)
    return (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)


def select_fit(tr):
    """Fit the selector on dev and choose tau on out-of-fold dev scores (accuracy-maximising)."""
    X, u, grp = tr[BASE + EVF].to_numpy(), tr.u.to_numpy(), tr.group.to_numpy()
    oof = gate.oof_scores(X, u, grp, kind="benefit")
    cand = np.unique(np.quantile(oof, np.linspace(0, 1, 201)))
    acc = [np.where(oof >= t, tr.c_ev, tr.c_ne).mean() for t in cand]
    tau = float(cand[int(np.argmax(acc))])
    return benefit_model(X, u), tau, float(max(acc))


def select_hidden(tr, te, He_tr, He_te, Hn_tr, Hn_te, c_sel, n_boot):
    """PREREG v2d: selection in MMA-RAG's setting. Four-outcome logistic on [PCA-64(h_NE), PCA-64(h_EV)] (Selector-H),
    optionally + VGate-select's features (Selector-HF); tau maximises out-of-fold dev accuracy."""
    def pcs(Htr, Hte):
        sc = StandardScaler().fit(Htr)
        pca = PCA(n_components=64, random_state=0).fit(sc.transform(Htr))
        return pca.transform(sc.transform(Htr)), pca.transform(sc.transform(Hte))
    (n_tr, n_te), (e_tr, e_te) = pcs(Hn_tr, Hn_te), pcs(He_tr, He_te)
    o_tr = (2 * tr.c_ne + tr.c_ev).astype(int).to_numpy()  # 0 both wrong, 1 fixes, 2 breaks, 3 both right
    grp, cn, ce = te.group.to_numpy(), te.c_ne.to_numpy().astype(float), te.c_ev.to_numpy().astype(float)
    out = {}
    for name, Xtr, Xte in [("selector_h", np.hstack([n_tr, e_tr]), np.hstack([n_te, e_te])),
                           ("selector_hf", np.hstack([n_tr, e_tr, np.nan_to_num(tr[BASE + EVF].to_numpy())]),
                            np.hstack([n_te, e_te, np.nan_to_num(te[BASE + EVF].to_numpy())]))]:
        score = lambda m, X: (lambda p, c: (p[:, c.index(1)] if 1 in c else 0) - (p[:, c.index(2)] if 2 in c else 0))(
            m.predict_proba(X), list(m[-1].classes_))
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        oof = np.zeros(len(tr))  # (PCA is fit on all dev rows; it never sees outcome labels)
        from sklearn.model_selection import GroupKFold
        for a_, b_ in GroupKFold(n_splits=5).split(Xtr, o_tr, tr.group.to_numpy()):
            oof[b_] = score(mk().fit(Xtr[a_], o_tr[a_]), Xtr[b_])
        cand = np.unique(np.quantile(oof, np.linspace(0, 1, 201)))
        acc = [np.where(oof >= t, tr.c_ev, tr.c_ne).mean() for t in cand]
        tau = float(cand[int(np.argmax(acc))])
        take = score(mk().fit(Xtr, o_tr), Xte) >= tau
        c = np.where(take, ce, cn)
        out[name] = dict(acc=float(c.mean()), tau=tau, dev_oof_acc=float(max(acc)), frac_keep_a0=float((~take).mean()),
                         minus_always_95ci=gate.cluster_bootstrap(grp, lambda i: float(c[i].mean() - ce[i].mean()), n_boot=n_boot, alpha=0.05),
                         minus_vgate_select_95ci=gate.cluster_bootstrap(grp, lambda i: float(c[i].mean() - c_sel[i].mean()), n_boot=n_boot, alpha=0.05))
    return out


def hok(tr, te, s_te, n_boot):
    """H-OK (PREREG v2): VGate-post at the read rate that maximises its out-of-fold DEV accuracy, compared on the
    held-out split with never- and always-retrieve (paired cluster bootstrap)."""
    z = np.zeros(len(tr))
    oof = gate.oof_scores(tr[BASE].to_numpy(), tr.u.to_numpy(), tr.group.to_numpy(), kind="benefit")
    k = int(np.argmax(gate.curve(oof, tr.c_ne, tr.c_ev, z, z)[0]))  # first maximum = lowest read rate
    cn, ce, grp = te.c_ne.to_numpy().astype(float), te.c_ev.to_numpy().astype(float), te.group.to_numpy()
    at = lambda i: gate.curve(s_te[i], cn[i], ce[i], np.zeros(len(i)), np.zeros(len(i)))[0][k]
    return dict(read_rate=float(gate.GRID[k]), acc=float(at(np.arange(len(te)))),
                minus_never_95ci=gate.cluster_bootstrap(grp, lambda i: float(at(i) - cn[i].mean()), n_boot=n_boot, alpha=0.05),
                minus_always_95ci=gate.cluster_bootstrap(grp, lambda i: float(at(i) - ce[i].mean()), n_boot=n_boot, alpha=0.05))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_run", required=True)
    ap.add_argument("--test_run", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--datasets", default="evqa,infoseek")
    ap.add_argument("--train_split", default="dev")
    ap.add_argument("--test_split", default="confirm")
    ap.add_argument("--ev_hidden_run", default="", help="run folder with evidence-pass hidden states (J8, PREREG v2d)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    rep, parts = {}, []
    for d in a.datasets.split(","):
        tr = add_ev_features(A.load(a.train_run, a.data, d, a.model, a.train_split), a.train_run, d, a.model)
        te = add_ev_features(A.load(a.test_run, a.data, d, a.model, a.test_split), a.test_run, d, a.model)
        Htr, Hte = U.load_hidden(a.train_run, d, a.model), U.load_hidden(a.test_run, d, a.model)
        L = sorted(Htr)[len(Htr) // 2]
        hx_tr, hx_te = Htr[L].reindex(tr.index).to_numpy(), Hte[L].reindex(te.index).to_numpy()
        S = C.scores(tr, te, hx_tr, hx_te)  # preregistered v1 gates, for paired comparisons
        # --- VGate-full: [free + QPP + PCA-64 hidden], benefit target
        sc, pca = StandardScaler().fit(hx_tr), None
        pca = PCA(n_components=64, random_state=0).fit(sc.transform(hx_tr))
        f_tr = np.hstack([tr[BASE].to_numpy(), pca.transform(sc.transform(hx_tr))])
        f_te = np.hstack([te[BASE].to_numpy(), pca.transform(sc.transform(hx_te))])
        S["vgate_full"] = benefit_score(benefit_model(f_tr, tr.u.to_numpy()), f_te)
        # --- four-outcome probe (MMA-RAG's target, closed-book state only): 0 both wrong, 1 fixes, 2 breaks, 3 both right
        o_tr = (2 * tr.c_ne + tr.c_ev).astype(int).to_numpy()
        m4 = make_pipeline(StandardScaler(), PCA(n_components=64, random_state=0),
                           LogisticRegression(C=0.1, max_iter=3000)).fit(hx_tr, o_tr)
        p4, c4 = m4.predict_proba(hx_te), list(m4[-1].classes_)
        S["probe4"] = (p4[:, c4.index(1)] if 1 in c4 else 0) - (p4[:, c4.index(2)] if 2 in c4 else 0)
        # --- VGate-select
        msel, tau, dev_acc = select_fit(tr)
        s_te = benefit_score(msel, te[BASE + EVF].to_numpy())
        take_ev = s_te >= tau
        c_sel = np.where(take_ev, te.c_ev, te.c_ne).astype(float)
        grp = te.group.to_numpy()
        cn, ce = te.c_ne.to_numpy().astype(float), te.c_ev.to_numpy().astype(float)
        d_always = gate.cluster_bootstrap(grp, lambda i: float(c_sel[i].mean() - ce[i].mean()), n_boot=a.n_boot, alpha=0.05)
        d_never = gate.cluster_bootstrap(grp, lambda i: float(c_sel[i].mean() - cn[i].mean()), n_boot=a.n_boot, alpha=0.05)
        sw = (~take_ev) & (te.agree.to_numpy() == 0)  # switched back to a_0 where the two answers differ
        rep[d] = dict(n_test=len(te), hok=hok(tr, te, S["benefit_post"], a.n_boot),
                      acc_never=float(cn.mean()), acc_always=float(ce.mean()),
                      acc_oracle_select=float(np.maximum(cn, ce).mean()), acc_select=float(c_sel.mean()), tau=tau,
                      dev_oof_acc_select=dev_acc, select_minus_always_95ci=d_always, select_minus_never_95ci=d_never,
                      frac_switched=float(sw.mean()),
                      switch_precision=float(((cn > ce) & sw).sum() / max(((cn != ce) & sw).sum(), 1)),
                      nauc={k: gate.nauc(gate.curve(S[k], cn, ce, np.zeros(len(te)), np.zeros(len(te)))[0])
                            for k in ["error_post", "benefit_post", "factorised_post", "probe_error", "probe_benefit",
                                      "probe4", "vgate_full", "signal_qpp_nqc", "oracle"]})
        gs = pathlib.Path(a.test_run) / f"gate_scores__{d}__{a.model}__{a.test_split}.csv"  # written by confirm.py
        if gs.exists():  # per-question scores of the new gates, for the demo
            g = pd.read_csv(gs, index_col="qid")
            g.index = g.index.astype(str)
            g["score_vgate_full"], g["score_probe4"] = pd.Series(S["vgate_full"], te.index), pd.Series(S["probe4"], te.index)
            g["sel_score"], g["sel_take_ev"] = pd.Series(s_te, te.index), pd.Series(take_ev.astype(int), te.index)
            g.to_csv(gs)
        He = U.load_hidden(a.ev_hidden_run, d, a.model, cond="ev") if a.ev_hidden_run else {}
        if He:  # PREREG v2d
            e_tr, e_te = He[L].reindex(tr.index).to_numpy(), He[L].reindex(te.index).to_numpy()
            assert not (np.isnan(e_tr).any() or np.isnan(e_te).any()), "evidence-pass hidden states do not cover the split"
            j8 = U.read(a.ev_hidden_run, d, a.model, "ev")  # regenerated answers: agreement with the original EV pass
            agree = float(np.mean([U.norm(j8[q]["text"]) == U.norm(t) for q, t in zip(te.index, te.ev_text)]))
            rep[d]["select_hidden"] = dict(select_hidden(tr, te, e_tr, e_te, hx_tr, hx_te, c_sel, a.n_boot), ev_rerun_answer_agreement=agree)
        rep[d]["policies"] = C.summarize(te, {k: S[k] for k in ("vgate_full", "probe4")})  # same columns as Table 1
        parts.append((te, S))
    comps = [("vgate_full", "benefit_post"), ("vgate_full", "probe_benefit"), ("vgate_full", "factorised_post"),
             ("vgate_full", "error_post"), ("probe_benefit", "probe4"), ("probe4", "probe_error")]
    rep["pooled_dnauc_95ci"] = {f"{x} - {y}": C.paired(parts, x, y, n_boot=a.n_boot) for x, y in comps}
    out = pathlib.Path(a.test_run) / f"extend__{a.model}"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1, default=float))
    lines = [f"# Extensions (PREREG v2b): {a.model}, fit on {a.train_split}, test on {a.test_split}\n",
             "| dataset | never | always | **VGate-select** | select − always [95% CI] | oracle select | switched | switch precision |",
             "|---|---|---|---|---|---|---|---|"]
    for d in a.datasets.split(","):
        r = rep[d]; p, lo, hi = r["select_minus_always_95ci"]
        lines.append(f"| {d} | {100*r['acc_never']:.1f} | {100*r['acc_always']:.1f} | **{100*r['acc_select']:.1f}** | "
                     f"{100*p:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] | {100*r['acc_oracle_select']:.1f} | "
                     f"{100*r['frac_switched']:.1f}% | {100*r['switch_precision']:.0f}% |")
    lines += ["\n| dataset | VGate-post read rate (dev-chosen) | accuracy | − never [95% CI] | − always [95% CI] |", "|---|---|---|---|---|"]
    for d in a.datasets.split(","):
        h = rep[d]["hok"]
        lines.append(f"| {d} | {100*h['read_rate']:.0f}% | {100*h['acc']:.1f} | " + " | ".join(
            f"{100*v[0]:+.2f} [{100*v[1]:+.2f}, {100*v[2]:+.2f}]" for v in (h["minus_never_95ci"], h["minus_always_95ci"])) + " |")
    if any("select_hidden" in rep[d] for d in a.datasets.split(",")):
        lines += ["\n| dataset | selector (PREREG v2d) | accuracy | − always [95% CI] | − VGate-select [95% CI] | keeps a_0 | EV re-run agreement |", "|---|---|---|---|---|---|---|"]
        for d in a.datasets.split(","):
            for k, v in rep[d].get("select_hidden", {}).items():
                if isinstance(v, dict):
                    lines.append(f"| {d} | {k} | {100*v['acc']:.1f} | " + " | ".join(
                        f"{100*x[0]:+.2f} [{100*x[1]:+.2f}, {100*x[2]:+.2f}]" for x in (v["minus_always_95ci"], v["minus_vgate_select_95ci"]))
                        + f" | {100*v['frac_keep_a0']:.1f}% | {100*rep[d]['select_hidden']['ev_rerun_answer_agreement']:.1f}% |")
    lines += ["\n| pooled ΔnAUC | point | 95% lo | 95% hi |", "|---|---|---|---|"]
    lines += [f"| {k} | {100*v[0]:+.2f} | {100*v[1]:+.2f} | {100*v[2]:+.2f} |" for k, v in rep["pooled_dnauc_95ci"].items()]
    for d in a.datasets.split(","):
        lines.append(f"\n{d} nAUC: " + ", ".join(f"{k} {100*v:.2f}" for k, v in rep[d]["nauc"].items()))
    pathlib.Path(str(out) + ".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

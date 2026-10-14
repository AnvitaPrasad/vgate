"""What can a read-or-skip gate earn? An identity, a ceiling and a three-number estimate, checked against every gate score we have.

A gate reads a set R of queries. Its gain over never-retrieve is  h*TPR - m*FPR,  where h and m are the shares of queries
that retrieval helps and harms, TPR = P(read | help) and FPR = P(read | harm); queries whose answer does not change do not
matter. So what a score can earn is fixed by h, m and its ROC curve for help against harm (cost-sensitive ROC analysis:
Provost and Fawcett 2001; a classifier beats both trivial policies only inside its operating range, Drummond and Holte 2006).

  ceiling   min(h, m) * (2A - 1)        any score whose ROC curve is concave with area A: a bound
  estimate  G*(h, m, A) = max_x [ h * Phi(d' + Phi^-1(x)) - m * x ] - max(0, h - m),   d' = sqrt(2) * Phi^-1(A)
            a score with an equal-variance binormal ROC curve: not a bound (see `n_above_estimate` in the output)

  python scripts/gate_bound.py                 -> experiments/results/gate_bound.{json,md}
  python scripts/gate_bound.py --prospective   -> experiments/results/gate_bound__prospective.{json,md}   (PREREG v9, B1-B4)

EXPLORATORY (default mode): every held-out score below has been looked at before. "Observed" is the best lead a score
reaches with the threshold, and the sign of the score, chosen with the held-out labels (what it could earn at most);
"split" chooses both on one half of the images and applies them to the other half.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import lodo  # noqa: E402
from vgate import gate  # noqa: E402

R = ROOT / "experiments/runs"
SCORES = ["benefit_pre", "benefit_post", "probe_benefit", "vgate_full", "probe4", "error_pre", "error_post", "probe_error", "factorised_post",
          "signal_ent_mean", "signal_lp_min", "signal_qpp_nqc", "signal_qpp_top1_z"]

estimate, ceiling, needed_auroc = gate.gain_estimate, gate.gain_ceiling, gate.auroc_needed  # the library functions (src/vgate/gate.py)


def observed(score, u):
    """Best gain over the better fixed policy that thresholding `score` can reach (threshold chosen with the labels)."""
    o = np.argsort(-score, kind="stable")
    g = np.concatenate([[0.0], np.cumsum(u[o])]) / len(u)
    return float(g.max() - max(0.0, u.mean()))


def split_half(score, u, groups, seed=2027):
    """Lead over the better fixed policy when the sign and the threshold of `score` are chosen on one half of the image
    groups and applied to the other half (mean of the two directions)."""
    g = np.asarray(groups); ids = np.unique(g)
    half = np.isin(g, np.random.default_rng(seed).permutation(ids)[: len(ids) // 2])
    out = []
    for a, b in ((half, ~half), (~half, half)):
        nz = a & (u != 0)
        s = score if gate.auroc(score[nz], u[nz] > 0) >= 0.5 else -score
        o = np.argsort(-s[a], kind="stable")
        k = int(np.concatenate([[0.0], np.cumsum(u[a][o])]).argmax())  # read the k highest-scored queries of half a
        read = s[b] >= s[a][o][k - 1] if k else np.zeros(int(b.sum()), bool)
        out.append(float((u[b] * read).mean() - max(0.0, u[b].mean())))
    return float(np.mean(out))


def rates(u):
    return float((u == 1).mean()), float((u == -1).mean())


def cvqa_scores(model, run_dev, run_confirm):
    """CVQA, held-out split: u, the pre-read gate scores fitted on the development split, the after-reading score of PREREG
    v7c (first-token log-probability of the evidence reply minus the closed-book reply's; lowest when the evidence reply
    names no option) and the registered rules' read decisions. None when the runs are not there."""
    import cvqa_audit as CA
    from vgate import uncertainty as U
    run_dev, run_confirm = pathlib.Path(run_dev), pathlib.Path(run_confirm)
    if not (run_confirm / f"scores__cvqa__{model}.csv").exists() or not (run_dev / f"scores__cvqa__{model}.csv").exists():
        return None
    CA.MODEL = model
    data = R / "J14/data"
    tr, Htr, seg_tr = CA.load(run_dev, data, "dev"); te, Hte, seg_te = CA.load(run_confirm, data, "confirm")
    D, S = CA.decide(tr, Htr, te, Hte, seg_tr, seg_te)
    ne, ev = U.read(run_confirm, "cvqa", model, "ne"), U.read(run_confirm, "cvqa", model, "ev")
    p0 = np.array([ne[q]["tok_logprobs"][0] for q in te.index]); p1 = np.array([ev[q]["tok_logprobs"][0] for q in te.index])
    no = te.no_letter_ev.to_numpy()
    S = dict(S, after_reading=np.where(no, -50.0, p1 - p0))
    rules = dict(benefit_full=D["benefit_full"], benefit_post=D["benefit_post"], after_reading=(p1 > p0) & ~no)
    return te.u.to_numpy(), S, rules


def more():
    """CVQA (main reader, held-out): the pre-read gates and the after-reading score; published gates (PREREG v8)."""
    import analyze as A
    import cvqa_audit as CA
    import external_gates as EG
    out = {}
    cv = cvqa_scores(CA.MODEL, R / "J14", R / "J15")
    if cv:
        u, S, _ = cv
        h, m = rates(u)
        c = dict(help=round(100 * h, 1), harm=round(100 * m, 1), auroc_needed_for_1pt=round(needed_auroc(h, m), 3),
                 auroc_floor_for_1pt=round(needed_auroc(h, m, shape="any"), 3), scores={})
        for k in ("benefit_full", "benefit_post", "benefit_pre", "after_reading"):
            a_ = gate.auroc(S[k][u != 0], u[u != 0] > 0)
            c["scores"][k] = dict(auroc=round(a_, 3), observed=round(100 * observed(S[k], u), 2), estimate=round(100 * estimate(h, m, a_), 2),
                                  ceiling=round(100 * ceiling(h, m, a_), 2))
        out["cvqa"] = c
    pub = {}
    runs = lodo.MODELS[EG.MODEL]["confirm"]
    for name in ("chen", "urag"):
        for d in EG.DATASETS:
            G = EG.gate_rows(name, d)
            if not G:
                continue
            fr = A.load(R / runs[d], ROOT / "data", d, EG.MODEL, "confirm"); fr = fr[fr.index.isin(G)]
            sc = np.array([G[q][0] for q in fr.index]); u = fr.u.to_numpy()
            h, m = rates(u)
            a_ = gate.auroc(sc[u != 0], u[u != 0] > 0)
            pub[f"{name}/{d}"] = dict(auroc=round(a_, 3), needed=round(needed_auroc(h, m), 3), floor=round(needed_auroc(h, m, shape="any"), 3),
                                      estimate=round(100 * estimate(h, m, max(a_, 1 - a_)), 2), ceiling=round(100 * ceiling(h, m, max(a_, 1 - a_)), 2))
    if pub:
        out["published"] = pub
    return out


def main():
    rows, cells = [], []
    for model, runs in lodo.MODELS.items():
        for d, run in runs["confirm"].items():
            f = R / run / f"gate_scores__{d}__{model}__confirm.csv"
            if not f.exists():
                continue
            g = pd.read_csv(f); u = g.u.to_numpy()
            h, m = rates(u)
            cells.append(dict(model=model, benchmark=d, n=len(g), help=round(100 * h, 1), harm=round(100 * m, 1), oracle_headroom=round(100 * (min(h, m)), 1),
                              auroc_needed_for_1pt=round(needed_auroc(h, m), 3), auroc_floor_for_1pt=round(needed_auroc(h, m, shape="any"), 3)))
            for s in SCORES:
                if f"score_{s}" not in g:
                    continue
                sc = np.nan_to_num(g[f"score_{s}"].to_numpy(float))
                A = gate.auroc(sc[u != 0], u[u != 0] > 0)
                if A < 0.5:  # sign fixed with the labels (optimistic for raw signals)
                    sc, A = -sc, 1 - A
                rows.append(dict(model=model, benchmark=d, score=s, auroc=round(A, 3), observed=round(100 * observed(sc, u), 2),
                                 split=round(100 * split_half(sc, u, g.image_id.to_numpy()), 2),
                                 estimate=round(100 * estimate(h, m, A), 2), ceiling=round(100 * ceiling(h, m, A), 2)))
    extra = more()
    T, C = pd.DataFrame(rows), pd.DataFrame(cells)
    err = over = T.observed - T.estimate
    best = T.sort_values("auroc").groupby(["model", "benchmark"]).tail(1).merge(C, on=["model", "benchmark"])
    share = (T.observed / T.ceiling)[T.ceiling > 0.2]
    rep = dict(n_points=len(T), mean_abs_error=round(float(err.abs().mean()), 2), max_abs_error=round(float(err.abs().max()), 2), bias=round(float(err.mean()), 2),
               pearson=round(float(np.corrcoef(T.observed, T.estimate)[0, 1]), 3), max_observed=round(float(T.observed.max()), 2),
               max_auroc=round(float(T.auroc.max()), 3), needed_min=round(float(C.auroc_needed_for_1pt.min()), 2), needed_max=round(float(C.auroc_needed_for_1pt.max()), 2),
               within_half_point=round(100 * float((err.abs() <= 0.5).mean()), 1),
               n_above_estimate=int((over > 1e-9).sum()), n_above_estimate_by_quarter_point=int((over > 0.25).sum()), max_above_estimate=round(float(over.max()), 2),
               n_above_ceiling=int((T.observed > T.ceiling + 0.005).sum()), floor_min=round(float(C.auroc_floor_for_1pt.min()), 2),
               floor_max=round(float(C.auroc_floor_for_1pt.max()), 2), cells_best_above_floor=int((best.auroc > best.auroc_floor_for_1pt).sum()),
               ceiling_at_best_min=round(float(best.ceiling.min()), 2), ceiling_at_best_max=round(float(best.ceiling.max()), 2),
               share_of_ceiling_median=round(100 * float(share.median()), 0), share_of_ceiling_max=round(100 * float(share.max()), 0), share_of_ceiling_n=int(len(share)),
               split_mean_abs_error=round(float((T.split - T.estimate).abs().mean()), 2), split_bias=round(float((T.split - T.estimate).mean()), 2),
               split_max=round(float(T.split.max()), 2), split_mean=round(float(T.split.mean()), 2), observed_mean=round(float(T.observed.mean()), 2),
               estimate_mean=round(float(T.estimate.mean()), 2), ceiling_mean=round(float(T.ceiling.mean()), 2), cells=C.to_dict("records"), points=T.to_dict("records"), **extra)
    L = ["# What a read-or-skip gate can earn: a ceiling and a three-number estimate against every held-out score (exploratory)", "",
         f"{rep['n_points']} scores (benchmark x reader x gate or signal). Observed = best lead over the better fixed policy that the score reaches with threshold and sign "
         f"chosen on the held-out labels; split = both chosen on one half of the images and applied to the other. Ceiling = min(h, m)(2A - 1), a bound for any concave ROC curve; "
         f"estimate = G*(h, m, A), an equal-variance binormal ROC curve.", "",
         f"**Estimate.** Mean absolute difference to the observed lead {rep['mean_abs_error']} points (largest {rep['max_abs_error']}; mean signed difference {rep['bias']:+.2f}; "
         f"{rep['within_half_point']}% within half a point); correlation {rep['pearson']}. It is not a bound: {rep['n_above_estimate']} of {rep['n_points']} observed leads exceed it, "
         f"{rep['n_above_estimate_by_quarter_point']} by more than 0.25 points, at most by {rep['max_above_estimate']}. Against the split-half lead: mean absolute difference "
         f"{rep['split_mean_abs_error']}, mean signed difference {rep['split_bias']:+.2f} (mean lead: observed {rep['observed_mean']}, split {rep['split_mean']}, estimate {rep['estimate_mean']}; "
         f"largest split-half lead {rep['split_max']}).", "",
         f"**Ceiling.** Exceeded by {rep['n_above_ceiling']} of {rep['n_points']}. A one-point lead is out of reach below an AUROC of {rep['floor_min']}-{rep['floor_max']} (by cell); the "
         f"best score of a cell is above that floor in {rep['cells_best_above_floor']} of {len(C)} cells, where the ceiling is {rep['ceiling_at_best_min']}-{rep['ceiling_at_best_max']} points. "
         f"Observed leads reach {rep['share_of_ceiling_median']:.0f}% of the ceiling in the median and at most {rep['share_of_ceiling_max']:.0f}% ({rep['share_of_ceiling_n']} scores "
         f"with a ceiling above 0.2 points).", "",
         f"Largest observed lead {rep['max_observed']}; largest help-vs-harm AUROC {rep['max_auroc']}; AUROC that the estimate needs for +1 point {rep['needed_min']}-{rep['needed_max']}.", "",
         "| reader | benchmark | helps % | harms % | AUROC floor for +1 point | AUROC needed (estimate) | best AUROC of any score | observed / split / estimate / ceiling |", "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        t = T[(T.model == c["model"]) & (T.benchmark == c["benchmark"])].sort_values("auroc").iloc[-1]
        L.append(f"| {c['model']} | {c['benchmark']} | {c['help']} | {c['harm']} | {c['auroc_floor_for_1pt']} | {c['auroc_needed_for_1pt']} | {t.auroc} ({t.score}) | "
                 f"{t.observed} / {t.split} / {t.estimate} / {t.ceiling} |")
    if "cvqa" in rep:
        c_ = rep["cvqa"]
        L += ["", f"CVQA (main reader, held-out): helps {c_['help']}%, harms {c_['harm']}%; AUROC floor for +1 point {c_['auroc_floor_for_1pt']}, needed at the estimate "
              f"{c_['auroc_needed_for_1pt']}. " + "; ".join(f"{k}: AUROC {v['auroc']}, observed best {v['observed']:+.2f}, estimate {v['estimate']:+.2f}, ceiling {v['ceiling']:+.2f}"
                                                         for k, v in c_["scores"].items()) + "."]
    if "published" in rep:
        L += ["", "Published gates (main reader, held-out), help-vs-harm AUROC of the gate's score / floor / needed at the estimate: "
              + "; ".join(f"{k} {v['auroc']} / {v['floor']} / {v['needed']}" for k, v in rep["published"].items()) + "."]
    out = ROOT / "experiments/results/gate_bound"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1)); out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(T.groupby("score")[["auroc", "observed", "split", "estimate", "ceiling"]].mean().round(2).to_string())


def prospective(a):
    """PREREG v9 (and its note v9a). B1: ReflectiVA's retrieval token (J17), per benchmark, the main reader's outcomes.
    B2-B3: a second reader on CVQA (J18), held-out split. B4: the ceiling on both. Tolerances are the registered ones."""
    import analyze as A
    import external_gates as EG
    rep, L = {}, ["# The gate estimate and ceiling on runs made after they were registered (PREREG v9, v9a)", ""]
    b1, runs = {}, lodo.MODELS[EG.MODEL]["confirm"]
    for d in EG.DATASETS:
        G = EG.gate_rows(a.b1_gate, d)
        if not G:
            continue
        fr = A.load(R / runs[d], ROOT / "data", d, EG.MODEL, "confirm"); fr = fr[fr.index.isin(G)]
        sc = np.array([G[q][0] for q in fr.index]); dec = np.array([G[q][1] for q in fr.index]); u = fr.u.to_numpy()
        h, m = rates(u)
        a_ = gate.auroc(sc[u != 0], u[u != 0] > 0)
        lead, est, need = float((u * dec).mean() - max(0.0, u.mean())), estimate(h, m, a_), needed_auroc(h, m)
        b1[d] = dict(n=len(fr), help=round(100 * h, 1), harm=round(100 * m, 1), auroc=round(a_, 3), needed=round(need, 3), floor=round(needed_auroc(h, m, shape="any"), 3),
                     lead_at_published_decision=round(100 * lead, 2), estimate=round(100 * est, 2), ceiling=round(100 * ceiling(h, m, a_), 2),
                     best_lead=round(100 * observed(sc, u), 2), B1_at_most_estimate_plus_0p3=bool(lead <= est + 0.003),
                     B1_below_one_point_where_auroc_below_needed=(bool(lead < 0.01) if a_ < need else None),
                     B4_best_lead_at_most_ceiling=bool(observed(sc, u) <= ceiling(h, m, a_) + 5e-5))
    rep["B1"] = b1 or f"NOT_RUN: no outputs of the gate '{a.b1_gate}' ({EG.GATES[a.b1_gate][1]})"
    L += [f"## B1 — {EG.GATES[a.b1_gate][0]} ({EG.GATES[a.b1_gate][1]})", ""] + ([f"- {d}: {json.dumps(v)}" for d, v in b1.items()] if b1 else [rep["B1"]]) + [""]
    cv = cvqa_scores(a.cvqa_model, a.cvqa_run_dev, a.cvqa_run_confirm)
    if cv:
        u, S, rules = cv
        h, m = rates(u)
        need, b2 = needed_auroc(h, m), {}
        for k in ("benefit_full", "benefit_post", "after_reading"):
            a_ = gate.auroc(S[k][u != 0], u[u != 0] > 0)
            obs, est, rule = observed(S[k], u), estimate(h, m, a_), float((u * rules[k]).mean() - max(0.0, u.mean()))
            b2[k] = dict(auroc=round(a_, 3), best_lead=round(100 * obs, 2), estimate=round(100 * est, 2), ceiling=round(100 * ceiling(h, m, a_), 2),
                         registered_rule_lead=round(100 * rule, 2), B2_best_lead_within_0p5_of_estimate=bool(abs(obs - est) <= 0.005),
                         B2_rule_at_most_estimate_plus_0p3=bool(rule <= est + 0.003),
                         B3_prediction=("undecided" if abs(a_ - need) <= 0.02 else "leads by one point" if a_ > need else "does not"),
                         B3_best_lead_at_least_one_point=bool(obs >= 0.01), B4_best_lead_at_most_ceiling=bool(obs <= ceiling(h, m, a_) + 5e-5))
            b2[k]["B3_correct"] = None if b2[k]["B3_prediction"] == "undecided" else bool((b2[k]["B3_prediction"] == "leads by one point") == (obs >= 0.01))
        rep["B2_B3"] = dict(model=a.cvqa_model, n=len(u), help=round(100 * h, 1), harm=round(100 * m, 1), needed=round(need, 3),
                            floor=round(needed_auroc(h, m, shape="any"), 3), scores=b2)
        L += [f"## B2, B3 — {a.cvqa_model} on CVQA (held-out, n = {len(u)}; helps {100 * h:.1f}%, harms {100 * m:.1f}%; AUROC needed {need:.3f})", ""]
        L += [f"- {k}: {json.dumps(v)}" for k, v in b2.items()]
    else:
        rep["B2_B3"] = f"NOT_RUN: no {a.cvqa_model} outputs under {a.cvqa_run_confirm}"
        L += ["## B2, B3 — a second reader on CVQA (J18)", "", rep["B2_B3"]]
    out = pathlib.Path(a.out) / "gate_bound__prospective"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1)); out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prospective", action="store_true", help="PREREG v9: the registered tests on J17 and J18")
    ap.add_argument("--b1_gate", default="reflectiva", help="the registered gate; another name only for a dry run of the code")
    ap.add_argument("--cvqa_model", default="Qwen2.5-VL-3B-Instruct"); ap.add_argument("--cvqa_run_dev", default=str(R / "J18"))
    ap.add_argument("--cvqa_run_confirm", default=str(R / "J18")); ap.add_argument("--out", default=str(ROOT / "experiments/results"))
    args = ap.parse_args()
    prospective(args) if args.prospective else main()

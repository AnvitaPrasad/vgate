"""Labels, out-of-fold gates, total-cost frontiers, paired ΔnAUC with image-cluster bootstrap.

Cost model (per query, tokens by default; GPU-seconds as a secondary unit):
  never-retrieve  = NE pass
  always-retrieve = EV pass only (it never needs the NE answer)
  gate g          = passes its features need (NE + e.g. samples) + EV pass for the queries it retrieves
Accuracy of a gate at retrieval rate b: retrieve the top-b fraction by score, answer with EV there, NE elsewhere.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

GRID = np.linspace(0, 1, 101)


def oof_scores(X, y, groups, kind="binary", n_splits=5, C=1.0, seed=0, pca=0):
    """Out-of-fold scores, GroupKFold by image. kind='binary' -> P(y=1); 'benefit' -> P(help) - P(harm).
    pca>0 adds PCA(pca) before the classifier (hidden-state probes), fit on training folds only."""
    X = np.nan_to_num(np.asarray(X, dtype=float))
    y = np.asarray(y)
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        steps = [StandardScaler()] + ([PCA(n_components=min(pca, len(tr) - 1), random_state=seed)] if pca else [])
        m = make_pipeline(*steps, LogisticRegression(C=C, max_iter=2000))
        if len(np.unique(y[tr])) < 2:
            out[te] = y[tr].mean()
            continue
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[te])
        cls = list(m.classes_)
        if kind == "binary":
            out[te] = p[:, cls.index(1)]
        else:  # expected accuracy gain from retrieving
            out[te] = (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)
    return out


def oof_scores_by_segment(X, u, groups, segment, n_splits=5, min_n=200):
    """Out-of-fold benefit scores of one gate per segment, each fitted on that segment's queries only. A segment with fewer
    than `min_n` queries, or without both a help and a harm case, keeps the score of the gate fitted on all traffic."""
    X, u, groups, seg = np.nan_to_num(np.asarray(X, dtype=float)), np.asarray(u), np.asarray(groups), np.asarray(segment).astype(str)
    out = oof_scores(X, u, groups, kind="benefit", n_splits=n_splits)
    for s_ in sorted(set(seg)):
        k = seg == s_
        if k.sum() >= min_n and (u[k] > 0).any() and (u[k] < 0).any() and len(set(groups[k])) >= n_splits:
            out[k] = oof_scores(X[k], u[k], groups[k], kind="benefit", n_splits=n_splits)
    return out


def audit(X, c_ne, c_ev, groups, segment, n_splits=5, decision=None, n_boot=0, alpha=0.05, conf_ne=None, conf_ev=None, fine=None, min_segment=3):
    """What has a gate learned? Read/skip policies on a log with a traffic `segment` per query (e.g. the benchmark or
    the source of the query). Returns a DataFrame: policy, macro accuracy over segments, share of queries read.
      fixed per segment      the better of never / always on each segment (needs the segment label)
      segment predicted      predict the segment from the signals, apply that segment's better fixed policy
      error / benefit gate   one gate for all traffic, no segment identity: read iff P(wrong) > .5 / P(help) > P(harm)
      gate per segment       a benefit gate fitted on each segment's own queries (needs the segment label): does
                             per-query gating add anything to knowing the segment?
      unseen segment         the same two gates fitted on the other segments only
      logged gate            (with `decision`, the 0/1 read decisions of any gate, e.g. a published one) those decisions
                             as logged, and replaced by the gate's most frequent decision on each segment
      after reading          (with `conf_ne`, `conf_ev`, a confidence of each reply) answer with the more confident reply;
                             this reads the evidence for every query and pays for both passes
    A log without signal columns (X with no column) gets every row that needs no fitted model.
    Everything fitted is scored out of fold (GroupKFold by `groups`) or on the left-out segment. With n_boot > 0 the table
    also gives each policy's difference to the fixed-per-segment policy (single segment: to the better of never / always)
    with a cluster-bootstrap interval over `groups`, resampled within segments and conditional on the fitted decisions.
    Many small segments (`fine`; by default more than 20 of them, e.g. the question type or template of each query): the
    fixed-per-segment policy and a logged gate's majority decision are estimated on the training folds (a segment with fewer
    than `min_segment` training queries gets the fold's better fixed policy), the unseen-segment gates are fitted on folds
    that keep whole segments out, accuracy is the mean over queries, and the segment classifier is left out."""
    X = np.nan_to_num(np.asarray(X, dtype=float))
    c0, c1, seg, groups = np.asarray(c_ne, float), np.asarray(c_ev, float), np.asarray(segment).astype(str), np.asarray(groups)
    u, wrong, names = (c1 - c0).astype(int), (1 - c0).astype(int), sorted(set(np.asarray(segment).astype(str)))
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))

    def benefit(m, F):
        p, cls = m.predict_proba(F), list(m.classes_)
        return (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)
    better = lambda idx: {s: bool(c1[idx][seg[idx] == s].mean() > c0[idx][seg[idx] == s].mean()) for s in names if (seg[idx] == s).any()}
    fit = X.ndim == 2 and X.shape[1] > 0  # no signal column: nothing to fit
    fine = len(names) > 20 if fine is None else bool(fine)
    dec = np.asarray(decision).astype(bool) if decision is not None else None
    keys = ("error gate", "benefit gate", "unseen segment: error gate", "unseen segment: benefit gate")
    if not fine:  # with a few large segments: the segment classifier, and one benefit gate per segment
        keys = ("segment predicted",) + keys[:2] + (("benefit gate per segment (needs the label)",) if len(names) > 1 else ()) + keys[2:]
    read = {k: np.zeros(len(u), bool) for k in keys} if fit else {}
    told, told_major = np.zeros(len(u), bool), np.zeros(len(u), bool)  # fine: per-segment policies from the training folds
    for tr, te in GroupKFold(n_splits=n_splits).split(u, u, groups) if fit or fine else []:
        if fine:
            t = pd.DataFrame(dict(s=seg[tr], d=c1[tr] - c0[tr], r=dec[tr] if dec is not None else False)).groupby("s").agg(d=("d", "mean"), r=("r", "mean"), n=("d", "size"))
            t = t[t.n >= min_segment]
            pol_d, pol_r = dict(zip(t.index, t.d > 0)), dict(zip(t.index, t.r > 0.5))
            told[te] = [pol_d.get(s, c1[tr].mean() > c0[tr].mean()) for s in seg[te]]
            told_major[te] = [pol_r.get(s, dec[tr].mean() > 0.5) for s in seg[te]] if dec is not None else False
        if not fit:
            continue
        if not fine:
            b = better(tr)
            pred = lr().fit(X[tr], seg[tr]).predict(X[te]) if len(set(seg[tr])) > 1 else seg[te]
            read["segment predicted"][te] = [b.get(s, True) for s in pred]
        read["error gate"][te] = lr().fit(X[tr], wrong[tr]).predict_proba(X[te])[:, 1] > 0.5 if len(set(wrong[tr])) > 1 else True
        read["benefit gate"][te] = benefit(lr().fit(X[tr], u[tr]), X[te]) > 0 if len(set(u[tr])) > 1 else True
    if fit and not fine and len(names) > 1:
        read["benefit gate per segment (needs the label)"] = oof_scores_by_segment(X, u, groups, seg, n_splits=n_splits) > 0
    unseen = GroupKFold(n_splits=n_splits).split(u, u, seg) if fine else ((seg != s_, seg == s_) for s_ in names)
    for tr, te in unseen if len(names) > 1 and fit else []:
        read["unseen segment: error gate"][te] = lr().fit(X[tr], wrong[tr]).predict_proba(X[te])[:, 1] > 0.5 if len(set(wrong[tr])) > 1 else True
        read["unseen segment: benefit gate"][te] = benefit(lr().fit(X[tr], u[tr]), X[te]) > 0 if len(set(u[tr])) > 1 else True
    if not fine:
        b_all = better(np.arange(len(u)))
        told = np.array([b_all[s] for s in seg])
    pol = {"never retrieve": np.zeros(len(u), bool), "always retrieve": np.ones(len(u), bool),
           "fixed per segment (needs the label)": told, **read, "oracle gate": u > 0}
    if decision is not None:
        if not fine:
            major = {s: bool(dec[seg == s].mean() > 0.5) for s in names}
            told_major = np.array([major[s] for s in seg])
        oracle = pol.pop("oracle gate")  # keep the oracle as the last row
        pol = {**pol, "logged gate (decision column)": dec, "logged gate: its majority decision per segment": told_major, "oracle gate": oracle}
    if conf_ne is not None and conf_ev is not None:
        oracle = pol.pop("oracle gate")
        pol = {**pol, "after reading: the more confident answer": np.asarray(conf_ev, float) > np.asarray(conf_ne, float), "oracle gate": oracle}
    if len(names) < 2:
        pol = {k: v for k, v in pol.items() if "segment" not in k}
    macro = (lambda r: float(np.where(r, c1, c0).mean())) if fine else (lambda r: float(np.mean([np.where(r, c1, c0)[seg == s].mean() for s in names])))
    reads_all = {"after reading: the more confident answer"}  # these read the evidence for every query, whichever answer they give
    out = pd.DataFrame([dict(policy=k, accuracy=round(100 * macro(r), 1), read=100.0 if k in reads_all else round(100 * float(r.mean()), 0))
                        for k, r in pol.items()])
    if n_boot:
        ref = pol.get("fixed per segment (needs the label)", pol["always retrieve"] if c1.mean() > c0.mean() else pol["never retrieve"])
        a_ref, code = np.where(ref, c1, c0), np.unique(seg, return_inverse=True)[1]

        def diff(r):
            d = np.where(r, c1, c0) - a_ref
            if fine:
                return cluster_bootstrap(groups, lambda i: float(d[i].mean()), n_boot=n_boot, alpha=alpha)
            return cluster_bootstrap(groups, lambda i: float(np.mean([d[i][code[i] == j].mean() for j in range(len(names))])),
                                     n_boot=n_boot, alpha=alpha, strata=code)
        ci = [diff(r) for r in pol.values()]
        out["vs_fixed"], out["lo"], out["hi"] = [[round(100 * v[j], 1) for v in ci] for j in range(3)]
    return out


def gain_ceiling(h, m, auroc):
    """Largest lead over the better fixed policy, as a fraction of queries, that a read-or-skip gate can reach when
    retrieval helps a share h and harms a share m of the queries and the gate's score separates help from harm with the
    given AUROC: min(h, m) * (2 * AUROC - 1), for any score whose ROC curve is concave.
    A gate that reads a set of queries gains h*TPR - m*FPR over never-retrieve (TPR, FPR: the shares of help and of harm
    cases it reads); queries whose answer does not change do not matter. Its lead over the better fixed policy is
    therefore at most min(h, m) * (TPR - FPR), and on a concave curve TPR - FPR <= 2 * AUROC - 1."""
    return min(h, m) * max(2 * float(auroc) - 1, 0.0)


_GRID = None


def gain_estimate(h, m, auroc):
    """The lead over the better fixed policy that a score of the given help-vs-harm AUROC reaches at its best threshold
    when its ROC curve is equal-variance binormal (d' = sqrt(2) * Phi^-1(AUROC)). An estimate for scores of that shape,
    not a bound: a score with a cleaner top or bottom than the binormal curve earns more, up to gain_ceiling.
    Cost-sensitive ROC analysis with help and harm as the two classes."""
    from scipy.stats import norm
    global _GRID
    if _GRID is None:
        x = np.linspace(1e-4, 1 - 1e-4, 4001)
        _GRID = (x, norm.ppf(x))
    x, z = _GRID
    d = np.sqrt(2) * norm.ppf(min(max(float(auroc), 0.5), 0.9999))
    best = float(np.max(h * norm.cdf(d + z) - m * x))
    return max(best, 0.0, h - m) - max(0.0, h - m)


def auroc_needed(h, m, target=0.01, shape="binormal"):
    """Help-vs-harm AUROC with which a gate leads the better fixed policy by `target` (nan: out of reach).
    shape="binormal": for a score with an equal-variance binormal ROC curve (gain_estimate), the usual shape;
    shape="any": the floor below which no score with a concave ROC curve reaches the target (gain_ceiling)."""
    if shape == "any":
        a = 0.5 + target / (2 * min(h, m)) if min(h, m) > 0 else float("nan")
        return a if a <= 1 else float("nan")
    lo, hi = 0.5, 0.9999
    if gain_estimate(h, m, hi) < target:
        return float("nan")
    for _ in range(24):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if gain_estimate(h, m, mid) >= target else (mid, hi)
    return hi


def labels_needed(h, m, confidence=0.95):
    """Paired queries (both answers judged) needed to tell which fixed policy is better with the given one-sided
    confidence, by the normal approximation to the mean of u in {-1, 0, +1}; inf when h == m."""
    from scipy.stats import norm
    if h == m:
        return float("inf")
    z = norm.ppf(confidence)
    return float(np.ceil(z ** 2 * (h + m - (h - m) ** 2) / (h - m) ** 2))


def null_lead(h, m, n):
    """What the best threshold of a score with NO signal leads the better fixed policy by, on a log of n queries of which
    a share h is helped and a share m harmed (a fraction; multiply by 100 for points). A score independent of the outcome
    orders the queries at random; the cumulative gain along that order is a random path of P = hn up-steps and Q = mn
    down-steps, and by the reflection principle P(max >= b) = C(P+Q, Q+b) / C(P+Q, P) for b above max(0, P-Q). The
    expectation of the maximum minus max(0, P-Q), summed exactly. This is the noise floor of a lead whose threshold was
    chosen on the labels it is scored on; with the sign of the score chosen too it is at most twice this."""
    from scipy.special import gammaln
    P, Q = int(round(h * n)), int(round(m * n))
    if P == 0 or Q == 0:
        return 0.0
    N, base = P + Q, max(0, P - Q)
    b = np.arange(base + 1, P + 1)
    logc = lambda k: gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
    return float(np.exp(logc(Q + b) - logc(P)).sum() / n)


def plan(c_ne, c_ev, segment=None, score=None, target=0.01, n_boot=0, seed=0, score_segment=None):
    """"Should you gate?" One row per traffic segment and one for all queries pooled: the outcome rates, the better fixed
    policy and the paired labels needed to find it, the oracle headroom, and the help-vs-harm AUROC a gate needs to lead
    that policy by `target`: auroc_floor (no score with a concave ROC curve gets there below it) and auroc_needed (a score
    of the usual, binormal shape). With n_boot > 0, needed_lo and needed_hi give the 5-95% range of auroc_needed when the
    two rates are re-drawn from a log of this size. With a score (e.g. a gate's out-of-fold predicted benefit): the AUROC it
    has there, the ceiling for that AUROC, the estimate for a score of the usual shape and the lead that reading where
    the score is positive did reach (points). `score_segment` (e.g. oof_scores_by_segment: one gate per segment) replaces
    `score` in the segment rows. A score that mostly recognises the segment has a high pooled AUROC and a low one within
    segments."""
    c0, c1 = np.asarray(c_ne, float), np.asarray(c_ev, float)
    u = c1 - c0
    seg = np.asarray(segment).astype(str) if segment is not None else np.full(len(u), "all")
    rng = np.random.default_rng(seed)
    rows = []
    for name in sorted(set(seg)) + (["all queries pooled"] if len(set(seg)) > 1 else []):
        k = np.ones(len(u), bool) if name == "all queries pooled" else seg == name
        n = int(k.sum())
        h, m = float((u[k] > 0).mean()), float((u[k] < 0).mean())
        r = dict(segment=name, n=n, help=round(100 * h, 1), harm=round(100 * m, 1), better_fixed="always" if h > m else "never",
                 labels_needed=labels_needed(h, m), headroom=round(100 * min(h, m), 1), null_lead=round(100 * null_lead(h, m, n), 2),
                 auroc_floor=round(auroc_needed(h, m, target, "any"), 2), auroc_needed=round(auroc_needed(h, m, target), 2))
        if n_boot:
            d = rng.multinomial(n, [h, m, max(0.0, 1 - h - m)], size=n_boot) / n
            a = np.array([auroc_needed(x[0], x[1], target) for x in d])
            a = np.where(np.isnan(a), 1.0, a)  # out of reach in that draw
            r.update(needed_lo=round(float(np.quantile(a, 0.05)), 2), needed_hi=round(float(np.quantile(a, 0.95)), 2))
        sc = score_segment if score_segment is not None and name != "all queries pooled" else score
        if sc is not None and (u[k] > 0).any() and (u[k] < 0).any():
            nz, sc = k & (u != 0), np.asarray(sc, float)
            a_ = auroc(sc[nz], u[nz] > 0)
            r.update(auroc=round(a_, 2), ceiling=round(100 * gain_ceiling(h, m, a_), 2), estimate=round(100 * gain_estimate(h, m, a_), 2),
                     lead=round(100 * float((u[k] * (sc[k] > 0)).mean() - max(0.0, h - m)), 2))
        rows.append(r)
    return pd.DataFrame(rows)


def export_gate(X, c_ne, c_ev, names):
    """A benefit gate fitted on a whole log, as plain numbers (JSON-serialisable): standardise each signal, take the
    multinomial logits over u in {-1, 0, +1}, read iff P(u=+1) > P(u=-1). `apply_gate` evaluates it without scikit-learn."""
    X = np.nan_to_num(np.asarray(X, dtype=float))
    u = (np.asarray(c_ev) - np.asarray(c_ne)).astype(int)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, u)
    sc, lr = m[0], m[-1]
    return dict(kind="vgate benefit gate", rule="read iff P(u=+1) > P(u=-1)", features=list(names), mean=sc.mean_.tolist(),
                scale=sc.scale_.tolist(), classes=[int(c) for c in lr.classes_], coef=lr.coef_.tolist(), intercept=lr.intercept_.tolist(),
                n=int(len(u)), help=float((u == 1).mean()), harm=float((u == -1).mean()))


def apply_gate(g, X):
    """Predicted benefit P(u=+1) - P(u=-1) of an exported gate for rows X (columns in the order of g['features'])."""
    z = (np.nan_to_num(np.asarray(X, dtype=float)) - np.array(g["mean"])) / np.array(g["scale"])
    coef, b = np.array(g["coef"]), np.array(g["intercept"])
    if coef.shape[0] == 1:  # two outcome classes in the log: scikit-learn keeps one row
        coef, b = np.vstack([-coef[0] / 2, coef[0] / 2]), np.array([-b[0] / 2, b[0] / 2])
    lg = z @ coef.T + b
    pr = np.exp(lg - lg.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
    cls = list(g["classes"])
    return (pr[:, cls.index(1)] if 1 in cls else 0) - (pr[:, cls.index(-1)] if -1 in cls else 0)


def curve(score, c_ne, c_ev, base_cost, ev_cost, grid=GRID, rng=None):
    """Accuracy and total cost at each retrieval rate b in grid (ties broken randomly but reproducibly)."""
    n = len(score)
    rng = rng or np.random.default_rng(0)
    order = np.lexsort((rng.random(n), -np.asarray(score)))
    c_ne, c_ev = np.asarray(c_ne, float)[order], np.asarray(c_ev, float)[order]
    ev_cost = np.asarray(ev_cost, float)[order]
    gain_cum = np.concatenate([[0], np.cumsum(c_ev - c_ne)])
    cost_cum = np.concatenate([[0], np.cumsum(ev_cost)])
    k = np.round(grid * n).astype(int)
    acc = (c_ne.sum() + gain_cum[k]) / n
    cost = (np.sum(base_cost) + cost_cum[k]) / n
    return acc, cost


def nauc(acc, grid=GRID, lo=0.0):
    """Normalised area under accuracy-vs-read-rate over [lo, 1] (lo>0: partial AUC, mean accuracy in that regime)."""
    m = grid >= lo - 1e-12
    return float(np.trapezoid(acc[m], grid[m]) / (1 - lo))


def cost_at_accuracy(acc, cost, target):
    """Smallest mean cost at which the curve reaches `target` accuracy (np.nan if never)."""
    ok = np.where(acc >= target - 1e-12)[0]
    return float(cost[ok[0]]) if len(ok) else np.nan


def cluster_bootstrap(groups, stat, n_boot=2000, seed=0, alpha=0.2, strata=None):
    """Resample image groups with replacement (within each stratum, e.g. dataset, when `strata` is given);
    stat(idx) -> float. Returns (point, lo, hi)."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    if strata is None:
        blocks = [np.arange(len(uniq))]
    else:
        g_stratum = np.asarray(strata)[[m[0] for m in members]]
        blocks = [np.where(g_stratum == s)[0] for s in np.unique(g_stratum)]
    rng = np.random.default_rng(seed)
    point = stat(np.arange(len(groups)))
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([b[rng.integers(0, len(b), len(b))] for b in blocks])
        vals.append(stat(np.concatenate([members[g] for g in pick])))
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def auroc(score, y):
    """Mann-Whitney AUROC (ties = 0.5); nan if one class is empty."""
    score, y = np.asarray(score, float), np.asarray(y).astype(bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def outcome_table(c_ne, c_ev):
    """Help / harm / both-right / both-wrong rates -- the four quadrants of retrieval."""
    c_ne, c_ev = np.asarray(c_ne), np.asarray(c_ev)
    return dict(help=float(np.mean((c_ne == 0) & (c_ev == 1))), harm=float(np.mean((c_ne == 1) & (c_ev == 0))),
                both_right=float(np.mean((c_ne == 1) & (c_ev == 1))),
                both_wrong=float(np.mean((c_ne == 0) & (c_ev == 0))),
                acc_never=float(c_ne.mean()), acc_always=float(c_ev.mean()),
                acc_oracle_gate=float(np.maximum(c_ne, c_ev).mean()))

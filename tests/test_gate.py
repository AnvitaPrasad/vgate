"""Exact-number tests for labels, frontiers and AUROC (synthetic)."""
import numpy as np

from vgate import gate


def test_outcome_table_quadrants():
    t = gate.outcome_table([1, 1, 0, 0], [1, 0, 1, 0])
    assert (t["help"], t["harm"], t["both_right"], t["both_wrong"]) == (0.25, 0.25, 0.25, 0.25)
    assert t["acc_oracle_gate"] == 0.75


def test_curve_endpoints_and_oracle_order():
    c_ne = np.array([1, 0, 0, 1]); c_ev = np.array([0, 1, 0, 1])
    base = np.full(4, 10.0); ev = np.full(4, 100.0)
    acc, cost = gate.curve(c_ev - c_ne, c_ne, c_ev, base, ev, grid=np.array([0, .25, .5, .75, 1]))
    assert acc[0] == 0.5 and acc[-1] == 0.5          # never / always
    assert acc[1] == 0.75                             # oracle order retrieves the helped query first
    assert cost[0] == 10.0 and cost[-1] == 110.0      # gate pays NE for all, EV for retrieved


def test_nauc_constant():
    assert abs(gate.nauc(np.full(101, 0.4)) - 0.4) < 1e-12


def test_cost_at_accuracy():
    acc = np.array([.2, .3, .5]); cost = np.array([1., 2., 3.])
    assert gate.cost_at_accuracy(acc, cost, .3) == 2.0 and np.isnan(gate.cost_at_accuracy(acc, cost, .9))


def test_auroc_perfect_and_ties():
    assert gate.auroc([3, 2, 1, 0], [1, 1, 0, 0]) == 1.0
    assert gate.auroc([1, 1, 1, 1], [1, 0, 1, 0]) == 0.5


def test_oof_benefit_ranks_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    u = np.where(x > 0.8, 1, np.where(x < -0.8, -1, 0))
    s = gate.oof_scores(x[:, None], u, groups=np.arange(600) // 3, kind="benefit")
    assert gate.auroc(s, u == 1) > 0.9 and gate.auroc(-s, u == -1) > 0.9


def test_cluster_bootstrap_contains_point():
    y = np.r_[np.ones(50), np.zeros(50)]
    p, lo, hi = gate.cluster_bootstrap(np.arange(100) // 2, lambda i: y[i].mean(), n_boot=200)
    assert lo <= p <= hi and p == 0.5


def test_audit_regime_structure():
    """Two segments with opposite regimes and signals that reveal only the segment: a gate matches the per-segment
    fixed policy, cannot beat it, and fails on a segment it was not fitted on."""
    rng = np.random.default_rng(0)
    n = 600
    seg = np.repeat(["helps", "hurts"], n)
    c0 = np.r_[rng.random(n) < 0.2, rng.random(n) < 0.7].astype(int)
    c1 = np.r_[rng.random(n) < 0.6, rng.random(n) < 0.4].astype(int)
    X = np.c_[(seg == "hurts") + 0.3 * rng.standard_normal(2 * n), rng.standard_normal(2 * n)]
    t = gate.audit(X, c0, c1, np.arange(2 * n), seg).set_index("policy").accuracy
    assert t["fixed per segment (needs the label)"] > max(t["never retrieve"], t["always retrieve"]) + 5
    assert abs(t["benefit gate"] - t["fixed per segment (needs the label)"]) < 3
    assert abs(t["segment predicted"] - t["fixed per segment (needs the label)"]) < 3
    assert t["unseen segment: benefit gate"] < t["fixed per segment (needs the label)"] - 5
    assert t["oracle gate"] >= t.drop("oracle gate").max()


def test_audit_logged_gate():
    """A logged gate whose decisions follow the segment plus noise: the audit shows that its per-segment majority does better."""
    rng = np.random.default_rng(1)
    n = 800
    seg = np.repeat(["helps", "hurts"], n)
    c0 = np.r_[rng.random(n) < 0.2, rng.random(n) < 0.7].astype(int)
    c1 = np.r_[rng.random(n) < 0.6, rng.random(n) < 0.4].astype(int)
    dec = np.r_[rng.random(n) < 0.85, rng.random(n) < 0.15].astype(int)
    t = gate.audit(rng.standard_normal((2 * n, 2)), c0, c1, np.arange(2 * n), seg, decision=dec).set_index("policy")
    logged, major = t.accuracy["logged gate (decision column)"], t.accuracy["logged gate: its majority decision per segment"]
    assert major > logged and abs(major - t.accuracy["fixed per segment (needs the label)"]) < 1e-9
    assert 45 < t.read["logged gate (decision column)"] < 55
    assert "logged gate (decision column)" not in set(gate.audit(rng.standard_normal((2 * n, 2)), c0, c1, np.arange(2 * n), seg).policy)


def test_audit_intervals():
    """With n_boot the table carries the difference to the fixed-per-segment policy and its interval."""
    rng = np.random.default_rng(2)
    n = 500
    seg = np.repeat(["helps", "hurts"], n)
    c0 = np.r_[rng.random(n) < 0.2, rng.random(n) < 0.7].astype(int)
    c1 = np.r_[rng.random(n) < 0.6, rng.random(n) < 0.4].astype(int)
    t = gate.audit(rng.standard_normal((2 * n, 2)), c0, c1, np.arange(2 * n), seg, n_boot=200).set_index("policy")
    fixed, never = t.loc["fixed per segment (needs the label)"], t.loc["never retrieve"]
    assert (fixed.vs_fixed, fixed.lo, fixed.hi) == (0.0, 0.0, 0.0)
    assert never.hi < 0 and never.lo <= never.vs_fixed <= never.hi
    assert abs(never.vs_fixed - (never.accuracy - fixed.accuracy)) < 0.11
    assert t.loc["oracle gate"].lo > 0


def test_export_and_apply_gate():
    """A gate exported as plain numbers gives the same predicted benefit as the fitted scikit-learn model."""
    import json
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(3)
    X = rng.standard_normal((600, 4))
    c0 = (rng.random(600) < 0.5).astype(int)
    c1 = (rng.random(600) < 1 / (1 + np.exp(-X[:, 0]))).astype(int)
    g = json.loads(json.dumps(gate.export_gate(X, c0, c1, ["a", "b", "c", "d"])))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, c1 - c0)
    pr, cls = m.predict_proba(X), list(m[-1].classes_)
    assert np.allclose(gate.apply_gate(g, X), pr[:, cls.index(1)] - pr[:, cls.index(-1)], atol=1e-6)
    assert g["features"] == ["a", "b", "c", "d"] and g["classes"] == [-1, 0, 1]


def test_audit_without_signals_and_after_reading():
    """A log with outcomes, a segment and a gate's decisions only (no signal column) still gets the rows that need no fit;
    confidences of the two replies add the after-reading row."""
    rng = np.random.default_rng(4)
    n = 400
    seg = np.repeat(["a", "b"], n)
    c0 = (rng.random(2 * n) < 0.5).astype(int); c1 = (rng.random(2 * n) < 0.5).astype(int)
    dec = (rng.random(2 * n) < 0.3).astype(int)
    t = gate.audit(np.zeros((2 * n, 0)), c0, c1, np.arange(2 * n), seg, decision=dec, n_boot=50)
    assert list(t.policy) == ["never retrieve", "always retrieve", "fixed per segment (needs the label)", "logged gate (decision column)",
                              "logged gate: its majority decision per segment", "oracle gate"]
    conf1 = np.where(c1 > c0, 0.9, 0.2) + 0.01 * rng.random(2 * n)  # a confidence that knows when the evidence answer is the better one
    t = gate.audit(np.zeros((2 * n, 0)), c0, c1, np.arange(2 * n), seg, conf_ne=np.full(2 * n, 0.5), conf_ev=conf1).set_index("policy")
    assert t.accuracy["after reading: the more confident answer"] == t.accuracy["oracle gate"]
    assert t.read["after reading: the more confident answer"] == 100.0  # it reads the evidence for every query


def test_gain_estimate_ceiling_and_needed_auroc():
    """The estimate is zero for a useless score, equals the oracle headroom for a perfect one, grows with the AUROC and
    never exceeds the ceiling; auroc_needed inverts both. A simulated binormal score earns the estimate; a score with a
    clean block of help cases at the top earns the ceiling, which is more."""
    h, m = 0.10, 0.15
    assert gate.gain_estimate(h, m, 0.5) < 1e-6 and abs(gate.gain_estimate(h, m, 0.99995) - min(h, m)) < 2e-3
    assert gate.gain_estimate(h, m, 0.6) < gate.gain_estimate(h, m, 0.7) < gate.gain_estimate(h, m, 0.8)
    assert all(gate.gain_estimate(h, m, a) <= gate.gain_ceiling(h, m, a) + 1e-9 for a in np.linspace(0.5, 1, 26))
    assert all(gate.gain_estimate(m, h, a) <= gate.gain_ceiling(m, h, a) + 1e-9 for a in np.linspace(0.5, 1, 26))
    assert abs(gate.gain_ceiling(h, m, 1.0) - min(h, m)) < 1e-12 and gate.gain_ceiling(h, m, 0.4) == 0
    a = gate.auroc_needed(h, m, 0.01)
    assert abs(gate.gain_estimate(h, m, a) - 0.01) < 5e-4 and 0.5 < a < 1
    f = gate.auroc_needed(h, m, 0.01, "any")
    assert abs(gate.gain_ceiling(h, m, f) - 0.01) < 1e-12 and 0.5 < f < a
    assert np.isnan(gate.auroc_needed(0.004, 0.5, 0.01)) and np.isnan(gate.auroc_needed(0.004, 0.5, 0.01, "any"))  # help is rarer than the target
    rng = np.random.default_rng(5)
    n = 200_000
    u = rng.choice([1, -1, 0], size=n, p=[h, m, 1 - h - m])
    lead = lambda s: (np.concatenate([[0], np.cumsum(u[np.argsort(-s)])]) / n).max() - max(0.0, u.mean())
    s = rng.standard_normal(n) + np.where(u == 1, 1.0, 0.0)  # binormal separation of help from harm; neutral queries get noise
    A = gate.auroc(s[u != 0], u[u != 0] > 0)
    assert abs(lead(s) - gate.gain_estimate(h, m, A)) < 2e-3
    t = rng.random(n) + np.where((u == 1) & (rng.random(n) < 0.3), 2.0, 0.0)  # 30% of the help cases sit alone at the top
    B = gate.auroc(t[u != 0], u[u != 0] > 0)
    assert abs(lead(t) - gate.gain_ceiling(h, m, B)) < 3e-3 and lead(t) > gate.gain_estimate(h, m, B) + 0.01
    assert gate.labels_needed(0.33, 0.05) < gate.labels_needed(0.14, 0.05) < gate.labels_needed(0.08, 0.14)
    seg = np.where(np.arange(n) < n // 2, "a", "b")
    r = gate.plan((u < 0).astype(int), (u > 0).astype(int), segment=seg, score=s).set_index("segment")  # c_ne = 1 on harm, c_ev = 1 on help
    assert list(r.index) == ["a", "b", "all queries pooled"] and r.better_fixed["all queries pooled"] == "never"
    assert abs(r.auroc["all queries pooled"] - round(A, 2)) < 1e-9 and abs(r.estimate["all queries pooled"] - round(100 * gate.gain_estimate(h, m, A), 2)) < 0.02
    assert (r.auroc_floor < r.auroc_needed).all() and (r.estimate <= r.ceiling).all()
    q = gate.plan((u[:300] < 0).astype(int), (u[:300] > 0).astype(int), n_boot=50).iloc[0]  # a log of 300 queries: the needed AUROC is uncertain
    assert q.needed_lo <= q.auroc_needed <= q.needed_hi and q.needed_hi - q.needed_lo > 0.02


def test_audit_with_many_small_segments():
    """Sixty question types of 100 queries; retrieval helps in half of the types and harms in the other half, and the signal
    names the type. A gate fitted on the log has learned the types it has seen: the policy that is told the type (estimated
    on the training folds) does as well, and on types kept out of the fit the gate is back at the fixed policies."""
    rng = np.random.default_rng(11)
    k, per = 60, 100
    seg = np.repeat(np.arange(k), per)
    helps = seg < k // 2
    u = np.where(helps, rng.choice([1, 0, -1], k * per, p=[.3, .68, .02]), rng.choice([1, 0, -1], k * per, p=[.02, .68, .3]))
    c0, c1 = (u < 0).astype(int), (u > 0).astype(int)
    X = np.eye(k)[seg] + 0.1 * rng.standard_normal((k * per, k))
    t = gate.audit(X, c0, c1, np.arange(k * per), seg.astype(str), n_boot=50).set_index("policy")
    assert "segment predicted" not in t.index  # left out with many segments
    never, told, g, unseen = (t.accuracy[n] for n in ("never retrieve", "fixed per segment (needs the label)", "benefit gate", "unseen segment: benefit gate"))
    assert told > never + 10 and g > never + 10 and abs(g - told) < 4 and unseen < g - 8
    assert abs(t.vs_fixed["fixed per segment (needs the label)"]) < 1e-9 and t.vs_fixed["unseen segment: benefit gate"] < -8
    few = gate.audit(X, c0, c1, np.arange(k * per), (seg % 3).astype(str)).set_index("policy")  # three segments: the usual table
    assert "segment predicted" in few.index


def test_gate_per_segment_row_and_planner_lead():
    """Two segments in which the same signal means opposite things: one gate for all traffic cannot use it, a gate fitted
    per segment can, and the planner's segment rows take the per-segment score and report the lead it reached."""
    rng = np.random.default_rng(3)
    n = 3000
    seg = np.where(np.arange(n) < n // 2, "a", "b")
    x = rng.standard_normal(n)
    good = np.where(seg == "a", x > 0.8, x < -0.8)   # where retrieval helps; elsewhere it mostly harms a little
    u = np.where(good, rng.choice([1, 0], n, p=[.8, .2]), rng.choice([-1, 0], n, p=[.25, .75]))
    c0, c1, g = (u < 0).astype(int), (u > 0).astype(int), np.arange(n)
    X = x[:, None]
    t = gate.audit(X, c0, c1, g, seg).set_index("policy").accuracy
    assert t["benefit gate per segment (needs the label)"] > t["fixed per segment (needs the label)"] + 5
    assert t["benefit gate per segment (needs the label)"] > t["benefit gate"] + 5
    s_all, s_seg = gate.oof_scores(X, u, g, kind="benefit"), gate.oof_scores_by_segment(X, u, g, seg)
    r = gate.plan(c0, c1, segment=seg, score=s_all, score_segment=s_seg).set_index("segment")
    assert r.auroc["a"] > 0.9 and r.auroc["b"] > 0.9 and r.auroc["all queries pooled"] < 0.7
    assert r.lead["a"] > 5 and r.lead["b"] > 5 and (r.lead[["a", "b"]] <= r.ceiling[["a", "b"]] + 1e-9).all()
    # the pooled score is not monotone in the chance of help (its ROC curve is not concave), so the ceiling does not bind it
    assert r.lead["all queries pooled"] > r.ceiling["all queries pooled"]


def test_null_lead_matches_shuffles():
    """The exact null lead against shuffled outcome orders: balanced and unbalanced logs, small and large."""
    rng = np.random.default_rng(0)
    for h, m, n in [(0.075, 0.068, 600), (0.1, 0.1, 1000), (0.142, 0.046, 3637), (0.05, 0.25, 400)]:
        P, Q = int(round(h * n)), int(round(m * n))
        u = np.array([1] * P + [-1] * Q)
        sim = np.mean([(lambda c: c.max() - max(0, c[-1]))(np.concatenate([[0], np.cumsum(rng.permutation(u))])) for _ in range(4000)]) / n
        exact = gate.null_lead(h, m, n)
        assert abs(exact - sim) < max(0.06 * sim, 2e-4), (h, m, n, exact, sim)
    assert gate.null_lead(0.3, 0.0, 500) == 0.0
    assert gate.null_lead(0.1, 0.1, 1000) > 0.008 > gate.null_lead(0.1, 0.1, 100000)  # 0.84 points on 1,000 balanced queries; it shrinks with n

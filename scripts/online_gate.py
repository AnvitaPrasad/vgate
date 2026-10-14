"""Can the gate be learned in deployment from feedback on the shown answer only? (PREREG v3b; simulation on logged runs.)

  python scripts/online_gate.py --split confirm   -> experiments/results/online_gate__confirm.{json,md}
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
from vgate import uncertainty as U  # noqa: E402
from vgate.live import Gates  # noqa: E402

R, DATA, MODEL = ROOT / "experiments/runs", ROOT / "data", "Qwen3-VL-4B-Instruct"
RUNS = dict(dev=dict(evqa="J1", infoseek="J1", okvqa="J6"), confirm=dict(evqa="J4", infoseek="J4", okvqa="J6"))
RUNS_BY_MODEL = {MODEL: RUNS, "Qwen2.5-VL-3B-Instruct": {s: dict(evqa="J2", infoseek="J2", okvqa="J6") for s in ("dev", "confirm")},
                 "InternVL3_5-4B-HF": {s: dict(evqa="J5", infoseek="J5", okvqa="J11") for s in ("dev", "confirm")},  # PREREG v4c
                 "Qwen3-VL-8B-Instruct": {s: dict(evqa="J7", infoseek="J7", okvqa="J12") for s in ("dev", "confirm")}}
EPS, WARM, EVERY, C = 0.05, 100, 100, 0.05


def stream(split, g, model=MODEL):
    """Pooled questions: standardised VGate-full inputs (frozen, label-free dev statistics), outcomes, dataset, frozen-gate score."""
    X, c0, c1, ds, uf = [], [], [], [], []
    for d, run in RUNS_BY_MODEL[model][split].items():
        s = A.load(R / run, DATA, d, model, split)
        H = U.load_hidden(R / run, d, model)[g.layer].reindex(s.index).to_numpy()
        F = np.concatenate([np.nan_to_num(s[g.base].to_numpy(dtype=np.float64)), g._pcs(H)], -1)
        X.append((F - g.z["full_mean"]) / g.z["full_scale"]); uf.append(g.full(s[g.base].to_numpy(), H))
        c0.append(s.c_ne.to_numpy()); c1.append(s.c_ev.to_numpy()); ds += [d] * len(s)
    return np.vstack(X), np.concatenate(c0), np.concatenate(c1), np.array(ds), np.concatenate(uf)


def fit(X, y):
    return LogisticRegression(C=C, max_iter=300).fit(X, y) if len(y) >= 20 and y.min() != y.max() else None


def run(X, c0, c1, seed, both=False, eps=EPS):
    """One pass over the stream. Returns (correctness of the shown answers, read decisions), in stream order."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(c0)); X, c0, c1 = X[order], c0[order], c1[order]
    act, m0, m1 = np.zeros(len(c0), int), None, None
    for lo in range(0, len(c0), EVERY):
        hi = min(lo + EVERY, len(c0))
        coin, explore = rng.random(hi - lo) < 0.5, rng.random(hi - lo) < eps
        if lo < WARM or m0 is None or m1 is None:
            act[lo:hi] = coin
        else:
            greedy = m1.predict_proba(X[lo:hi])[:, 1] > m0.predict_proba(X[lo:hi])[:, 1]
            act[lo:hi] = np.where(explore, coin, greedy)
        seen0, seen1 = (slice(0, hi),) * 2 if both else (np.where(act[:hi] == 0)[0], np.where(act[:hi] == 1)[0])
        m0, m1 = fit(X[seen0], c0[seen0]), fit(X[seen1], c1[seen1])
    return np.where(act == 1, c1, c0), act, order


def running(got, block=500):
    """Accuracy per consecutive block of the stream: (questions seen, accuracy in that block)."""
    n = len(got) // block
    return block * (np.arange(n) + 1), got[: n * block].reshape(n, block).mean(1)


def summary(vals):
    v = 100 * np.asarray(vals)
    return [round(float(v.mean()), 2), round(float(v.min()), 2), round(float(v.max()), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="confirm")
    ap.add_argument("--orders", type=int, default=10)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    g = Gates(ROOT / "src/vgate/gates" / f"{a.model}.npz")
    X, c0, c1, ds, uf = stream(a.split, g, a.model)
    n, third = len(c0), len(c0) // 3
    fixed = np.concatenate([c1[ds == d] if c1[ds == d].mean() >= c0[ds == d].mean() else c0[ds == d] for d in RUNS[a.split]])
    ref = dict(n=n, never=round(100 * c0.mean(), 2), always=round(100 * c1.mean(), 2), fixed_by_dataset=round(100 * fixed.mean(), 2),
               oracle=round(100 * np.maximum(c0, c1).mean(), 2))
    rows = {k: dict(all=[], last=[], read=[]) for k in ("one_sided", "both_outcomes", "frozen_dev_gate", "always", "never")}
    for seed in range(a.orders):
        for name, kw in (("one_sided", {}), ("both_outcomes", dict(both=True))):
            got, act, order = run(X, c0, c1, seed, **kw)
            rows[name]["all"].append(got.mean()); rows[name]["last"].append(got[-third:].mean()); rows[name]["read"].append(act[-third:].mean())
        fz = np.where(uf > 0, c1, c0)[order]
        for name, v, rd in (("frozen_dev_gate", fz, (uf > 0)[order]), ("always", c1[order], np.ones(n)), ("never", c0[order], np.zeros(n))):
            rows[name]["all"].append(v.mean()); rows[name]["last"].append(v[-third:].mean()); rows[name]["read"].append(rd[-third:].mean())
    res = {k: {m: summary(v[m]) for m in v} for k, v in rows.items()}
    last = {k: np.array(v["last"]) for k, v in rows.items()}
    exp = dict(O1=bool(np.all(100 * (last["one_sided"] - last["always"]) >= 3.0)),
               O2=bool(abs(100 * (last["one_sided"] - last["both_outcomes"]).mean()) <= 1.5),
               O3=bool(100 * (last["frozen_dev_gate"] - last["one_sided"]).mean() <= 2.0))
    # exploratory: the 800 logged live-retriever queries (J9), cold start
    live = None
    lp = ROOT / "experiments/results/live_stream.npz"
    if lp.exists() and a.model == MODEL:  # the live logs are Qwen3-VL-4B's
        z = np.load(lp)
        Xl = (np.concatenate([np.nan_to_num(z["base"].astype(np.float64)), g._pcs(z["hidden"])], -1) - g.z["full_mean"]) / g.z["full_scale"]
        tr = np.where(g.full(z["base"], z["hidden"]) > 0, z["c1"], z["c0"])  # the dev-fitted gate, transferred to the live retriever
        outs = [run(Xl, z["c0"], z["c1"], s)[0] for s in range(a.orders)]
        live = dict(n=int(len(z["c0"])), never=round(100 * z["c0"].mean(), 2), always=round(100 * z["c1"].mean(), 2),
                    transferred_gate=round(100 * tr.mean(), 2), one_sided=summary([o.mean() for o in outs]),
                    one_sided_last_half=summary([o[len(o) // 2:].mean() for o in outs]))
    f = lambda v: f"{v[0]:.1f} ({v[1]:.1f}–{v[2]:.1f})"
    lines = [f"# Gate learned from one-sided feedback ({a.model}, {a.split} stream, n={n}, {a.orders} orders; PREREG v3b)", "",
             f"Micro accuracy on the pooled stream: never {ref['never']}, always {ref['always']}, better fixed policy per dataset "
             f"{ref['fixed_by_dataset']}, oracle {ref['oracle']}.", "", "| policy | whole stream | last third | read share (last third) |", "|---|---|---|---|"]
    names = dict(one_sided=f"online, shown answer judged only (ε={EPS})", both_outcomes="online, both outcomes judged",
                 frozen_dev_gate="VGate-full fitted on dev (frozen)", always="always", never="never")
    lines += [f"| {names[k]} | {f(v['all'])} | {f(v['last'])} | {f(v['read'])} |" for k, v in res.items()]
    lines += ["", "Mean (min–max) over orders; order variation only.", "", "## Expectations"] + [f"- {k}: {'held' if v else 'NOT held'}" for k, v in exp.items()]
    if live:
        lines += ["", f"Exploratory, live-retriever logs (n={live['n']}, micro accuracy, cold start): never {live['never']}, always {live['always']}, "
                  f"dev-fitted gate transferred {live['transferred_gate']}, online one-sided {f(live['one_sided'])} "
                  f"(second half of the stream {f(live['one_sided_last_half'])})."]
    out = ROOT / "experiments/results" / (f"online_gate__{a.split}" + ("" if a.model == MODEL else f"__{a.model}"))
    pathlib.Path(str(out) + ".json").write_text(json.dumps(dict(split=a.split, model=a.model, reference=ref, results=res, expectations=exp, live=live,
                                                        settings=dict(eps=EPS, warm=WARM, every=EVERY, C=C, orders=a.orders)), indent=1))
    pathlib.Path(str(out) + ".md").write_text("\n".join(lines) + "\n")  # model names contain dots: no with_suffix
    print("\n".join(lines))


if __name__ == "__main__":
    main()

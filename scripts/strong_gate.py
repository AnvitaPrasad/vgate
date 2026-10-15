"""PREREG v14 (registered, exploratory): a stronger gate — the full closed-book hidden state of all three saved layers.

Per reader and benchmark: G-lin (L2 logistic regression) and G-mlp (one hidden layer of 256 ReLU units, early stopping) on
[free + retriever-score features, standardised concatenation of the three layers' full hidden states], 3-class target u,
regularisation chosen by 5-fold image-grouped CV on the development questions (help-vs-harm AUROC). Rule: read iff
P(help) - P(harm) > 0.

  python scripts/strong_gate.py --phase forecast   -> experiments/results/strong_gate__forecast.{json,md} (development data only)
  python scripts/strong_gate.py --phase test       -> experiments/results/strong_gate.{json,md} (applies the saved gates once to held-out)
"""
import argparse
import json
import pathlib
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import lodo  # noqa: E402
import operating_point as OP  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

GRID = dict(lin=[1e-3, 1e-2, 1e-1], mlp=[1e-3, 1e-2, 1e-1])
OUT = ROOT / "experiments/results/strong_gate"
MODELS_DIR = ROOT / "build/strong_gate_models"  # fitted in the forecast phase, applied unchanged in the test phase


def model(kind, c):
    if kind == "lin":
        return LogisticRegression(C=c, max_iter=3000)
    return MLPClassifier(hidden_layer_sizes=(256,), alpha=c, early_stopping=True, random_state=0, max_iter=300)


def score(m, X):
    p, cls = m.predict_proba(X), list(m.classes_)
    return (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)


def features(fr, run, dataset, model_name):
    h = U.load_hidden(run, dataset, model_name)
    H = np.hstack([h[l].reindex(fr.index).to_numpy() for l in sorted(h)])
    return np.hstack([np.nan_to_num(fr[OP.BASE].to_numpy()), H])


def cells():
    for m, splits in lodo.MODELS.items():
        for d in ("evqa", "infoseek", "okvqa"):
            yield m, d, ROOT / "experiments/runs" / splits["dev"][d], ROOT / "experiments/runs" / splits["confirm"][d], ROOT / "data"
    yield "Qwen3-VL-4B-Instruct", "cvqa", ROOT / "experiments/runs/J14", ROOT / "experiments/runs/J15", ROOT / "experiments/runs/J14/data"


def fit(kind, X, u, g):
    best = None
    for c in GRID[kind]:
        s = np.zeros(len(u))
        for a, b in GroupKFold(5).split(X, u, g):
            sc = StandardScaler().fit(X[a]); s[b] = score(model(kind, c).fit(sc.transform(X[a]), u[a]), sc.transform(X[b]))
        auc = gate.auroc(s[u != 0], u[u != 0] > 0)
        if best is None or auc > best[1]:
            best = (c, auc)
    sc = StandardScaler().fit(X)
    return sc, model(kind, best[0]).fit(sc.transform(X), u), best


def run_cell(args):
    phase, (m, d, rdev, rte, data) = args
    key = f"{m}|{d}"; mf = MODELS_DIR / (key.replace("|", "__") + ".pkl")
    tr = A.load(rdev, data, d, m, "dev"); u = tr.u.to_numpy().astype(int)
    h, mm = float((u == 1).mean()), float((u == -1).mean())
    r = dict(n_dev=len(u), help_dev=round(100 * h, 1), harm_dev=round(100 * mm, 1), needed_dev=round(gate.auroc_needed(h, mm), 3), gates={})
    if phase == "forecast":
        Xtr = features(tr, rdev, d, m); g = tr.group.to_numpy(); r["dims"] = Xtr.shape[1]; fitted = {}
        for kind in GRID:
            sc, mod, (c, auc) = fit(kind, Xtr, u, g); fitted[kind] = (sc, mod)
            r["gates"][kind] = dict(reg=c, auroc_cv=round(auc, 3), forecast=round(100 * gate.gain_estimate(h, mm, auc), 2))
        MODELS_DIR.mkdir(parents=True, exist_ok=True); mf.write_bytes(pickle.dumps(fitted))
    else:
        fc = json.loads(pathlib.Path(str(OUT) + "__forecast.json").read_text())[key]; fitted = pickle.loads(mf.read_bytes())
        r["dims"] = fc["dims"]
        te = A.load(rte, data, d, m, "confirm"); Xte = features(te, rte, d, m)
        ut, gt = te.u.to_numpy().astype(int), te.group.to_numpy(); always = (ut == 1).mean() > (ut == -1).mean()
        for kind, (sc, mod) in fitted.items():
            s = score(mod, sc.transform(Xte)); dd = ut * (s > 0) - (ut if always else 0)
            pt, lo, hi = gate.cluster_bootstrap(gt, lambda i: float(dd[i].mean()), n_boot=2000, alpha=0.05)
            r["gates"][kind] = dict(fc["gates"][kind], n=len(ut), auroc=round(gate.auroc(s[ut != 0], ut[ut != 0] > 0), 3),
                                    read=round(100 * float((s > 0).mean()), 1), lead=[round(100 * x, 2) for x in (pt, lo, hi)])
    print(key, json.dumps(r), flush=True)
    return key, r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", choices=["forecast", "test"], required=True)
    ap.add_argument("--workers", type=int, default=3); a = ap.parse_args()
    with ProcessPoolExecutor(a.workers) as ex:
        rep = dict(ex.map(run_cell, [(a.phase, c) for c in cells()]))
    path = pathlib.Path(str(OUT) + ("__forecast" if a.phase == "forecast" else ""))
    path.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    L = [f"# PREREG v14 — a stronger gate (full hidden states of three layers): {a.phase}", "",
         "| reader | benchmark | dims | gate | reg | dev-CV AUROC | needed (dev) | forecast G* | held-out AUROC | read % | lead [95%] |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in rep.items():
        for kind, v in r["gates"].items():
            ld = f"{v['lead'][0]:+.2f} [{v['lead'][1]:+.2f}, {v['lead'][2]:+.2f}]" if "lead" in v else ""
            L.append(f"| {k.replace('|', ' | ')} | {r['dims']} | {kind} | {v['reg']} | {v['auroc_cv']} | {r['needed_dev']} | {v['forecast']:+.2f} | "
                     f"{v.get('auroc', '')} | {v.get('read', '')} | {ld} |")
    path.with_suffix(".md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()

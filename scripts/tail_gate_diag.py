"""Diagnosis of PREREG v10's outcome (EXPLORATORY, specified after the result): is the lead of a gate fitted on one benchmark's
own log a per-query prediction, or the identity of the question string?

Per benchmark of a reader's held-out outcomes (signals: the scalar pre-read signals and 64 hidden-state components, as in
demo/sample_log.csv), with a logistic and a gradient-boosted benefit model, rule: read iff P(help) - P(harm) > 0:
  image folds     five folds out of fold, image groups kept together (PREREG v10)
  string folds    five folds that keep every question string together
  told the string the better fixed policy of the query's question string, estimated on the training folds (image folds;
                  a string with fewer than `MIN` training questions gets the benchmark's better fixed policy)
  1,000 labels    image folds, each model fitted on 1,000 questions drawn from its training folds
  string folds, repeated   (benchmarks where at least a fifth of the questions use a string that occurs ten times or more)
                  REPEATS random assignments of the question strings to five folds of similar size: mean and range of the
                  lead and of the AUROC, because one assignment of a few large strings moves the result

  python scripts/tail_gate_diag.py   -> experiments/results/tail_gate_diag.{json,md}
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import tail_gate_check as T  # noqa: E402
from vgate import gate  # noqa: E402

M, RUNS, MIN, REPEATS = "Qwen3-VL-4B-Instruct", dict(evqa="J4", infoseek="J4", okvqa="J6"), 3, 10


def oof(make, X, u, groups, n_train=None, seed=0):
    out, rng = np.zeros(len(u)), np.random.default_rng(seed)
    for tr, te in GroupKFold(n_splits=5).split(X, u, groups):
        if n_train and len(tr) > n_train:
            tr = rng.choice(tr, n_train, replace=False)
        m = make().fit(X[tr], u[tr])
        p, cls = m.predict_proba(X[te]), list(m.classes_)
        out[te] = (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)
    return out


def string_folds_repeated(make, X, u, strings, repeats=REPEATS):
    """Lead of the rule and help-vs-harm AUROC with whole question strings held out, over `repeats` random assignments of
    the strings to five folds (largest strings first, ties in random order, each into the currently smallest fold)."""
    ids, cnt = np.unique(strings, return_counts=True)
    leads, aucs = [], []
    for seed in range(repeats):
        rng = np.random.default_rng(seed)
        order, size, fold = rng.permutation(len(ids)), np.zeros(5), {}
        for i in order[np.argsort(-cnt[order], kind="stable")]:
            j = int(np.argmin(size + rng.random(5) * 1e-6)); fold[ids[i]] = j; size[j] += cnt[i]
        fo, s = np.array([fold[x] for x in strings]), np.zeros(len(u))
        for j in range(5):
            tr, te = fo != j, fo == j
            m = make().fit(X[tr], u[tr])
            p, cls = m.predict_proba(X[te]), list(m.classes_)
            s[te] = (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)
        leads.append(100 * float((u * (s > 0)).mean() - max(0.0, u.mean()))); aucs.append(gate.auroc(s[u != 0], u[u != 0] > 0))
    return dict(repeats=repeats, lead_mean=round(float(np.mean(leads)), 2), lead_min=round(min(leads), 2), lead_max=round(max(leads), 2),
                auroc_mean=round(float(np.mean(aucs)), 3), auroc_min=round(min(aucs), 3), auroc_max=round(max(aucs), 3))


def told_string(u, strings, groups):
    """Read decisions of the policy that knows the question string: that string's better fixed policy on the training folds."""
    read = np.zeros(len(u), bool)
    for tr, te in GroupKFold(n_splits=5).split(u, u, groups):
        base = u[tr].mean() > 0
        t = pd.DataFrame(dict(s=strings[tr], u=u[tr])).groupby("s").u.agg(["mean", "size"])
        pol = {s: bool(r["mean"] > 0) for s, r in t.iterrows() if r["size"] >= MIN}
        read[te] = [pol.get(s, base) for s in strings[te]]
    return read


def strings_of(path, qids):
    recs = {r["qid"]: r for r in map(json.loads, open(path))}
    return np.array([" ".join(recs[q]["question"].lower().split()) for q in qids])


def cells(model, cvqa):
    """(benchmark, X, u, image groups, question strings) of one reader's held-out outcomes."""
    if cvqa:  # the fourth benchmark (main reader): signals as below, hidden-state components fitted on its development split
        import cvqa_audit as CA
        import operating_point as OP
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        CA.MODEL = model
        data = ROOT / "experiments/runs/J14/data"
        _, Hdev, _ = CA.load(ROOT / "experiments/runs/J14", data, "dev"); fr, H, _ = CA.load(ROOT / "experiments/runs/J15", data, "confirm")
        sc = StandardScaler().fit(Hdev); pcs = PCA(n_components=64, random_state=0).fit(sc.transform(Hdev))
        X = np.hstack([np.nan_to_num(fr[OP.BASE].to_numpy(float)), pcs.transform(sc.transform(H))])
        yield "cvqa", X, fr.u.to_numpy().astype(int), fr.group.to_numpy(), strings_of(data / "processed/cvqa.jsonl", fr.index)
    elif model == M:  # the main reader: the released sample log (PREREG v10's data)
        log = pd.read_csv(ROOT / "demo/sample_log.csv")
        feats = [c for c in log.select_dtypes("number") if c not in ("c_ne", "c_ev", "group", "segment", "cost_ne", "cost_ev", "decision")]
        for d, run in RUNS.items():
            fr = A.load(ROOT / "experiments/runs" / run, ROOT / "data", d, M, "confirm")
            k = log[log.segment == d]
            assert len(k) == len(fr) and (k.c_ne.to_numpy() == fr.c_ne.to_numpy().astype(int)).all() and (k.c_ev.to_numpy() == fr.c_ev.to_numpy().astype(int)).all()
            yield d, np.nan_to_num(k[feats].to_numpy(float)), (k.c_ev - k.c_ne).to_numpy().astype(int), k["group"].to_numpy(), strings_of(ROOT / "data/processed" / f"{d}.jsonl", fr.index)
    else:  # another reader: the same signals, built as scripts/make_sample_log.py builds them
        import lodo
        import operating_point as OP
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        _, Hdev = lodo.load(model, "dev"); F, H = lodo.load(model, "confirm")
        Hd = np.vstack(list(Hdev.values()))
        sc = StandardScaler().fit(Hd); pcs = PCA(n_components=64, random_state=0).fit(sc.transform(Hd))
        for d, fr in F.items():
            X = np.hstack([np.nan_to_num(fr[OP.BASE].to_numpy(float)), pcs.transform(sc.transform(H[d]))])
            yield d, X, fr.u.to_numpy().astype(int), fr.group.to_numpy(), strings_of(ROOT / "data/processed" / f"{d}.jsonl", fr.index)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=M); ap.add_argument("--cvqa", action="store_true", help="the fourth benchmark (main reader) instead of the three")
    a = ap.parse_args()
    tag = ("__cvqa" if a.cvqa else "") + ("" if a.model == M else f"__{a.model}")
    rep, L = {}, [f"# Diagnosis of PREREG v10: per-query prediction or the identity of the question string? (exploratory; {a.model}, held-out)", "",
                  f"Lead over the benchmark's better fixed policy in points, 95% image-cluster intervals; rule: read iff P(help) - P(harm) > 0. "
                  f"A question string needs {MIN} training questions to get its own policy.", ""]
    for d, X, u, g, strings in cells(a.model, a.cvqa):
        h, m = float((u == 1).mean()), float((u == -1).mean())
        n_str = len(set(strings)); cnt = pd.Series(strings).value_counts()
        cell = dict(n=len(u), help=round(100 * h, 1), harm=round(100 * m, 1), question_strings=n_str, auroc_floor=round(gate.auroc_needed(h, m, shape="any"), 3),
                    auroc_needed=round(gate.auroc_needed(h, m), 3), share_in_strings_of_10_or_more=round(100 * float(cnt[cnt >= 10].sum() / len(u)), 1), rows={})

        def lead(read):
            pt, lo, hi = gate.cluster_bootstrap(g, lambda i: float((u[i] * read[i]).mean() - max(0.0, u[i].mean())), n_boot=2000, alpha=0.05)
            return dict(lead=round(100 * pt, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2), read_rate=round(100 * float(read.mean()), 1))

        def row(name, s):
            a_ = gate.auroc(s[u != 0], u[u != 0] > 0)
            cell["rows"][name] = dict(lead(s > 0), auroc=round(a_, 3), estimate=round(100 * gate.gain_estimate(h, m, a_), 2), ceiling=round(100 * gate.gain_ceiling(h, m, a_), 2))
            return s > 0
        reads = {}
        for name, make in T.MODELS.items():
            reads[name] = row(f"{name}, image folds", oof(make, X, u, g))
            row(f"{name}, string folds", oof(make, X, u, strings))
            row(f"{name}, image folds, 1,000 labels", oof(make, X, u, g, n_train=1000))
        ts = told_string(u, strings, g)
        cell["rows"]["told the string (image folds)"] = lead(ts)
        for nm, r_ in reads.items():  # the gate against the policy that is told the string, paired
            pt, lo, hi = gate.cluster_bootstrap(g, lambda i: float((u[i] * r_[i]).mean() - (u[i] * ts[i]).mean()), n_boot=2000, alpha=0.05)
            cell[f"{nm}_minus_told_string"] = dict(diff=round(100 * pt, 2), lo=round(100 * lo, 2), hi=round(100 * hi, 2))
        if cell["share_in_strings_of_10_or_more"] >= 20:
            cell["string_folds_repeated"] = {name: string_folds_repeated(make, X, u, strings) for name, make in T.MODELS.items()}
        rep[d] = cell
        L += [f"## {d}: n = {cell['n']}, helps {cell['help']}%, harms {cell['harm']}%; AUROC floor {cell['auroc_floor']}, needed at the estimate {cell['auroc_needed']}; "
              f"{n_str} question strings, {cell['share_in_strings_of_10_or_more']}% of questions in strings used ten times or more", "",
              "| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |", "|---|---|---|---|---|---|"]
        L += [f"| {nm} | {v.get('auroc', '')} | {v['read_rate']} | {v['lead']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] | {v.get('estimate', '')} | {v.get('ceiling', '')} |" for nm, v in cell["rows"].items()]
        L += ["", "Gate minus told-the-string (image folds): " + "; ".join(
            f"{nm} {cell[nm + '_minus_told_string']['diff']:+.2f} [{cell[nm + '_minus_told_string']['lo']:+.2f}, {cell[nm + '_minus_told_string']['hi']:+.2f}]"
            for nm in ("logistic", "boosted")) + ".", ""]
        if "string_folds_repeated" in cell:
            L += [f"String folds over {REPEATS} random assignments: " + "; ".join(
                f"{nm} lead {v['lead_mean']:+.2f} on average ({v['lead_min']:+.2f} to {v['lead_max']:+.2f}), AUROC {v['auroc_mean']} ({v['auroc_min']}-{v['auroc_max']})"
                for nm, v in cell["string_folds_repeated"].items()) + ".", ""]
    out = ROOT / f"experiments/results/tail_gate_diag{tag}"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

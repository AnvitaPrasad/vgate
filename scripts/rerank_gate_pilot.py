"""PREREG v6e (PILOT, E-VQA development split, no model pass): which decision is worth a gate once the evidence is re-ranked?

Three decisions, labelled from one scoring pass (experiments/results/kbimage_reader_pass__questions.csv):
  A  skip or read the retriever's five (the paper's case, for reference);
  B  skip or read the image-re-ranked five;
  C  read the retriever's five or the re-ranked five.
For each: the shares of questions the action helps and harms, the planner's AUROC floor and needed AUROC for a one-point lead,
and gates fitted out of fold (five folds by image) on VGate-full's features (F0) and on those plus eight label-free
image-similarity features (F1), with the paper's logistic model and the boosted trees of PREREG v10.

  python scripts/rerank_gate_pilot.py --images <folder with thumbs.json + img/>   -> experiments/results/rerank_gate_pilot.{json,md}
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.parse

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import kbimage_pilot as KP  # noqa: E402
import kbimage_rerank_diag as RD  # noqa: E402
import lodo  # noqa: E402
import operating_point as OP  # noqa: E402
import tail_gate_check as T  # noqa: E402
from vgate import gate  # noqa: E402

N, K, W = RD.N, RD.K, RD.W
IMG = ["sim_max", "sim_retriever_first", "sim_reranked_first", "margin", "sim_std", "with_image", "changed", "z_retriever_first"]
DECISIONS = dict(A=("c_ne", "c_ev", "skip or read the retriever's five"), B=("c_ne", "c_rr", "skip or read the re-ranked five"),
                 C=("c_ev", "c_rr", "the retriever's five or the re-ranked five"))


def image_features(img, split):
    """Eight features of the photograph's similarity to the twenty candidates' lead images; none uses a label."""
    thumbs = json.loads((img / "thumbs.json").read_text())
    kb = {t: img / "img" / (urllib.parse.quote(t, safe="")[:150] + ".jpg") for t, u in thumbs.items() if u}
    kb = {t: p for t, p in kb.items() if p.exists()}
    ids = set((ROOT / f"data/splits/evqa_{split}.txt").read_text().split())
    raw = pd.read_parquet(ROOT / "data/raw/evqa_preflmr/data/test-00000-of-00001.parquet"); raw = raw[raw.question_id.isin(ids)].set_index("question_id")
    recs = {r["qid"]: r for r in map(json.loads, open(ROOT / "data/processed/evqa.jsonl")) if r["qid"] in ids}
    qimg = {f_: pathlib.Path(dp) / f_ for dp, _, fs in os.walk(ROOT / "data/raw/evqa_images") for f_ in fs}
    qpath = {i: qimg[pathlib.Path(recs[i]["image_file"]).name] for i in recs}
    E = RD.embeddings(img, sorted(set(kb.values()) | set(qpath.values())))
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    rows = {}
    for i in sorted(recs):
        L = list(raw.at[i, "retrieved_passage"])[:N]
        t = [KP.TITLE(x["passage_id"]) for x in L]; sc = np.array([x["score"] for x in L], float)
        qv = E[str(qpath[i])]
        sim = np.array([float(E[str(kb[x])] @ qv) if x in kb and str(kb[x]) in E else np.nan for x in t])
        cover = float(np.mean(~np.isnan(sim)))
        fill = np.nanmean(sim) if np.isfinite(sim).any() else 0.0
        sim = np.where(np.isnan(sim), fill, sim)
        best = {}
        for j, x in enumerate(t):
            best.setdefault(x, sc[j])
        zr = z(np.array([best[x] for x in t])); comb = zr + W * z(sim)
        order = np.lexsort((np.arange(N), -comb))
        page = pd.Series(comb, index=t).groupby(level=0).max().sort_values(ascending=False).to_numpy()  # one combined score per page
        rows[i] = dict(sim_max=float(sim.max()), sim_retriever_first=float(sim[0]), sim_reranked_first=float(sim[order[0]]),
                       margin=float(page[0] - page[1]) if len(page) > 1 else 0.0, sim_std=float(sim.std()), with_image=cover,
                       changed=float(len(set(order[:K]) - set(range(K)))), z_retriever_first=float(zr[0]))
    return pd.DataFrame.from_dict(rows, orient="index")[IMG]


def oof(make, base, H, extra, u, groups):
    """Out-of-fold P(help) - P(harm); the hidden state's scaler and PCA are fitted in fold."""
    out = np.zeros(len(u))
    for tr, te in GroupKFold(n_splits=5).split(base, u, groups):
        sc = StandardScaler().fit(H[tr]); pca = PCA(n_components=64, random_state=0).fit(sc.transform(H[tr]))
        X = lambda k: np.hstack([base[k], pca.transform(sc.transform(H[k]))] + ([extra[k]] if extra is not None else []))
        m = make().fit(X(tr), u[tr])
        p, cls = m.predict_proba(X(te)), list(m.classes_)
        out[te] = (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)
    return out


def frame(model, split, img):
    """Features, hidden states, paired outcomes and image groups of one reader on one split, aligned by question."""
    Fs, Hs = lodo.load(model, split)
    fr, H = Fs["evqa"], Hs["evqa"]
    q = pd.read_csv(ROOT / f"experiments/results/kbimage_reader_pass{'' if split == 'dev' else '__confirm'}__questions.csv")
    d = q[q.model == model].set_index("qid").reindex(fr.index)
    F = image_features(img, split).reindex(fr.index)
    assert d.c_ne.notna().all() and F.notna().all().all()
    return np.nan_to_num(fr[OP.BASE].to_numpy(float)), H, F.to_numpy(float), d, d.group.to_numpy()


def fitted(make, tr, te, use_images, u):
    """P(help) - P(harm) on `te` of a model fitted on all of `tr` (the paper's protocol: development -> held-out)."""
    sc = StandardScaler().fit(tr[1]); pca = PCA(n_components=64, random_state=0).fit(sc.transform(tr[1]))
    X = lambda f: np.hstack([f[0], pca.transform(sc.transform(f[1]))] + ([f[2]] if use_images else []))
    m = make().fit(X(tr), u)
    p, cls = m.predict_proba(X(te)), list(m.classes_)
    return (p[:, cls.index(1)] if 1 in cls else 0) - (p[:, cls.index(-1)] if -1 in cls else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="development images (thumbs.json + img/)")
    ap.add_argument("--split", default="dev", choices=["dev", "confirm"], help="confirm: PREREG v6f, gates fitted on development data and applied to the held-out split")
    ap.add_argument("--images_confirm", help="held-out images; needed with --split confirm")
    a = ap.parse_args()
    held, alpha = a.split == "confirm", (0.05 if a.split == "confirm" else 0.2)
    models = list(pd.read_csv(ROOT / "experiments/results/kbimage_reader_pass__questions.csv").model.unique())
    rep = {}
    for model in models:
        dev = frame(model, "dev", pathlib.Path(a.images))
        base, H, extra, d, g = frame(model, "confirm", pathlib.Path(a.images_confirm)) if held else dev
        rep[model] = {}
        for name, (c0, c1, what) in DECISIONS.items():
            u = (d[c1] - d[c0]).to_numpy().astype(int)
            u_dev = (dev[3][c1] - dev[3][c0]).to_numpy().astype(int)
            h, m = float((u == 1).mean()), float((u == -1).mean())
            cell = dict(decision=what, n=len(u), help=round(100 * h, 1), harm=round(100 * m, 1), better_fixed="take" if h > m else "leave",
                        oracle_lead=round(100 * min(h, m), 1), auroc_floor=round(gate.auroc_needed(h, m, shape="any"), 3),
                        auroc_needed=round(gate.auroc_needed(h, m), 3), fits={})
            if held:
                cell["dev"] = dict(help=round(100 * float((u_dev == 1).mean()), 1), harm=round(100 * float((u_dev == -1).mean()), 1))
                cell["rarer_share_unrounded"] = dict(held_out=round(100 * min(h, m), 3), dev=round(100 * float(min((u_dev == 1).mean(), (u_dev == -1).mean())), 3))
            if name == "C":  # where the switch harms: is the gold passage among both fives, lost, gained, or in neither?
                where = np.select([(d.gold_retriever == 1) & (d.gold_reranked == 1), (d.gold_retriever == 1) & (d.gold_reranked == 0),
                                   (d.gold_retriever == 0) & (d.gold_reranked == 1)], ["in both", "lost", "gained"], "in neither")
                cell["harm_by_gold_passage"] = {k: int(((u == -1) & (where == k)).sum()) for k in ("in both", "lost", "gained", "in neither")}
                cell["help_by_gold_passage"] = {k: int(((u == 1) & (where == k)).sum()) for k in ("in both", "lost", "gained", "in neither")}
            for feats, ex in (("F0", None), ("F1", extra)):
                for mname, make in T.MODELS.items():
                    s = fitted(make, dev[:3], (base, H, extra), ex is not None, u_dev) if held else oof(make, base, H, ex, u, g)
                    A_ = gate.auroc(s[u != 0], u[u != 0] > 0)
                    take = s > 0
                    pt, lo, hi = gate.cluster_bootstrap(g, lambda i: float((u[i] * take[i]).mean() - max(0.0, u[i].mean())), n_boot=2000, alpha=alpha)
                    cell["fits"][f"{feats} {mname}"] = dict(auroc=round(A_, 3), take_rate=round(100 * float(take.mean()), 1), lead=round(100 * pt, 2), lo=round(100 * lo, 2),
                                                            hi=round(100 * hi, 2), estimate=round(100 * gate.gain_estimate(h, m, A_), 2),
                                                            ceiling=round(100 * gate.gain_ceiling(h, m, A_), 2))
            rep[model][name] = cell
    main_, second = models[0], models[1]
    if held:  # PREREG v6f
        bc = [(mo, k) for mo in models for k in ("B", "C")]
        rep["F1"] = bool(all(abs(min(rep[mo][k]["help"], rep[mo][k]["harm"]) - min(rep[mo][k]["dev"]["help"], rep[mo][k]["dev"]["harm"])) <= 2.0 for mo, k in bc))
        # added after claim check 15: the registered line above compares shares rounded to one decimal; this is the same test unrounded
        rep["F1_unrounded"] = bool(all(abs(rep[mo][k]["rarer_share_unrounded"]["held_out"] - rep[mo][k]["rarer_share_unrounded"]["dev"]) <= 2.0 for mo, k in bc))
        rep["F1_largest_difference"] = round(max(abs(rep[mo][k]["rarer_share_unrounded"]["held_out"] - rep[mo][k]["rarer_share_unrounded"]["dev"]) for mo, k in bc), 3)
        rep["F2"] = bool(all(v["lead"] < 1.0 for mo, k in bc for v in rep[mo][k]["fits"].values()))
        rep["F3"] = bool(all(v["auroc"] < rep[mo]["C"]["auroc_floor"] for mo in models for v in rep[mo]["C"]["fits"].values()))
    else:
        rep["go"] = [f"{k} {f}" for k in ("B", "C") for f, v in rep[main_][k]["fits"].items()
                     if v["lead"] >= 1.0 and v["lo"] > 0 and rep[second][k]["fits"][f]["lead"] > 0]
    L = [("# PREREG v6f — the planner's forecast for two further decisions, on the held-out split (preregistered; E-VQA, gates fitted on development data; 95% image-cluster intervals)"
          if held else "# PREREG v6e — which decision is worth a gate once the evidence is re-ranked (pilot; E-VQA development split, n = 1,000; 80% image-cluster intervals)"), "",
         "A: skip or read the retriever's five. B: skip or read the re-ranked five. C: the retriever's five or the re-ranked five. "
         "F0: VGate-full's features; F1: F0 and eight image-similarity features. Rule: take the action iff P(help) - P(harm) > 0" + (", fitted on development data." if held else ", out of fold."), "",
         f"| reader | decision | helps % | harms % | oracle lead | AUROC floor / needed | fit | AUROC | lead [{100 - int(100 * alpha)}%] | estimate | ceiling |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in (main_, second):
        for k, c in rep[model].items():
            for f, v in c["fits"].items():
                L.append(f"| {model} | {k} | {c['help']} | {c['harm']} | {c['oracle_lead']} | {c['auroc_floor']} / {c['auroc_needed']} | {f} | {v['auroc']} | "
                         f"{v['lead']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}] | {v['estimate']:+.2f} | {v['ceiling']:+.2f} |")
    L += [""] + [f"Decision C, {model}: questions the switch harms / helps by where the gold passage is: "
                 + "; ".join(f"{k} {rep[model]['C']['harm_by_gold_passage'][k]} / {rep[model]['C']['help_by_gold_passage'][k]}" for k in rep[model]["C"]["harm_by_gold_passage"]) + "."
                 for model in (main_, second)]
    L += ["", ("Forecasts: " + "; ".join(f"{k} {'held' if rep[k] else 'NOT held'}" for k in ("F1", "F2", "F3")) + f". F1 without rounding the shares: {'held' if rep['F1_unrounded'] else 'NOT held'} (largest difference {rep['F1_largest_difference']} points; allowed 2.0).") if held else
          f"GO (a B or C fit of the main reader leads by a point with the interval above zero, positive on the second reader): {rep['go'] or 'none'}."]
    out = ROOT / ("experiments/results/rerank_gate_pilot" + ("__confirm" if held else ""))
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

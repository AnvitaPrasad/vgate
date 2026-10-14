"""Pilot (PREREG v6b, development data, no model pass): does re-ranking PreFLMR's top-20 with the candidates' page
images bring the gold page into the five passages that are read?

  python scripts/kbimage_rerank.py --images <folder with thumbs.json + img/>   -> experiments/results/kbimage_rerank.{json,md}

A page's score is z(best retriever score of its passages) + w * z(image similarity to the question's photograph), both
z-scored within the question's twenty candidates; DINOv2-base, one lead image per page (a page without one gets the
question's mean similarity). w is chosen among {0.5, 1, 2, 4} on four folds of image groups and applied to the fifth.
Passages are ordered by their page's score, then by retriever rank; the first five are the re-ranked evidence.
"""
import argparse
import json
import pathlib
import sys
import urllib.parse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import kbimage_pilot as KP  # noqa: E402
from vgate import gate  # noqa: E402

WEIGHTS, N, K = (0.5, 1.0, 2.0, 4.0), 20, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n_boot", type=int, default=2000)
    a = ap.parse_args()
    img = pathlib.Path(a.images)
    thumbs = json.loads((img / "thumbs.json").read_text())
    kb = {t: img / "img" / (urllib.parse.quote(t, safe="")[:150] + ".jpg") for t, u in thumbs.items() if u}
    kb = {t: p for t, p in kb.items() if p.exists()}
    dev = set((ROOT / "data/splits/evqa_dev.txt").read_text().split())
    raw = pd.read_parquet(ROOT / "data/raw/evqa_preflmr/data/test-00000-of-00001.parquet"); raw = raw[raw.question_id.isin(dev)].set_index("question_id")
    recs = {r["qid"]: r for r in map(json.loads, open(ROOT / "data/processed/evqa.jsonl")) if r["qid"] in dev}
    qimg = {}
    import os
    for dp, _, fs in os.walk(ROOT / "data/raw/evqa_images"):
        for f_ in fs:
            qimg[f_] = pathlib.Path(dp) / f_
    qids = sorted(recs)
    qpath = {i: qimg[pathlib.Path(recs[i]["image_file"]).name] for i in qids}
    E = {str(k): v for k, v in KP.embed("dinov2", sorted(set(kb.values()) | set(qpath.values()))).items()}
    T = KP.TITLE
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    rows = []
    for i in qids:
        L = list(raw.at[i, "retrieved_passage"])[:N]; gold = set(map(str, raw.at[i, "pos_item_ids"])); gt = {T(x) for x in gold}
        t = [T(x["passage_id"]) for x in L]; sc = np.array([x["score"] for x in L], float)
        qv = E[str(qpath[i])]
        sim = np.array([float(E[str(kb[x])] @ qv) if x in kb and str(kb[x]) in E else np.nan for x in t])
        cover = float(np.mean(~np.isnan(sim)))
        sim = np.where(np.isnan(sim), np.nanmean(sim) if np.isfinite(np.nanmean(sim)) else 0.0, sim)
        best = {}  # a page's retriever score is that of its best-ranked passage
        for j, x in enumerate(t):
            best.setdefault(x, sc[j])
        zs, zi = z(np.array([best[x] for x in t])), z(sim)
        rows.append(dict(qid=i, group=recs[i]["image_id"], t=t, zs=zs, zi=zi, cover=cover, page_gold=np.array([x in gt for x in t]),
                         psg_gold=np.array([x["passage_id"] in gold for x in L])))

    def top5(r, w):
        order = np.lexsort((np.arange(N), -(r["zs"] + w * r["zi"])))  # page score, then retriever rank
        return order[:K]

    def stats(r, idx):
        return dict(page_in5=bool(r["page_gold"][idx].any()), psg_in5=bool(r["psg_gold"][idx].any()), psg_first=bool(r["psg_gold"][idx[0]]),
                    page_first=bool(r["page_gold"][idx[0]]))
    base = pd.DataFrame([stats(r, np.arange(K)) for r in rows])
    groups = np.array([r["group"] for r in rows])
    re_, chosen = [None] * len(rows), []
    for tr, te in GroupKFold(n_splits=5).split(groups, groups=groups):
        w = max(WEIGHTS, key=lambda w_: np.mean([rows[j]["page_gold"][top5(rows[j], w_)].any() for j in tr]))
        chosen.append(w)
        for j in te:
            re_[j] = stats(rows[j], top5(rows[j], w))
    rer = pd.DataFrame(re_)
    rep = dict(n=len(rows), candidates=N, image_coverage=round(100 * float(np.mean([r["cover"] for r in rows])), 1), weights_chosen=chosen,
               gold_page_in_top20=round(100 * float(np.mean([r["page_gold"].any() for r in rows])), 1))
    for k in ("page_in5", "psg_in5", "page_first", "psg_first"):
        d = (rer[k].to_numpy().astype(float) - base[k].to_numpy().astype(float))
        v = gate.cluster_bootstrap(groups, lambda i: float(d[i].mean()), n_boot=a.n_boot, alpha=0.2)
        rep[k] = dict(retriever=round(100 * float(base[k].mean()), 1), reranked=round(100 * float(rer[k].mean()), 1), diff=[round(100 * x, 1) for x in v])
    rep["go"] = bool(rep["page_in5"]["diff"][0] >= 6.0 and rep["page_in5"]["diff"][1] > 3.0)
    names = dict(page_in5="gold page among the five read", psg_in5="gold passage among the five read", page_first="gold page ranked first", psg_first="gold passage ranked first")
    L = [f"# Re-ranking the top-20 with candidate page images (PREREG v6b pilot; E-VQA dev, n = {rep['n']}; 80% intervals)", "",
         f"Candidates with an image on disk: {rep['image_coverage']}%. Gold page among the top-20: {rep['gold_page_in_top20']}%. Weights chosen on the training folds: {chosen}.", "",
         "| % of questions | retriever order | re-ranked | difference |", "|---|---|---|---|"]
    L += [f"| {names[k]} | {rep[k]['retriever']} | {rep[k]['reranked']} | {rep[k]['diff'][0]:+.1f} [{rep[k]['diff'][1]:+.1f}, {rep[k]['diff'][2]:+.1f}] |" for k in names]
    L += ["", f"**Go / no-go (PREREG v6b): {'GO' if rep['go'] else 'NO-GO'}** (needs the gold page among the five read to rise by at least 6 points, interval above +3)."]
    out = ROOT / "experiments/results/kbimage_rerank"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1)); out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

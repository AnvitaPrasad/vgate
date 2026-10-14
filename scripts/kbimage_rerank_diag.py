"""Diagnosis of the PREREG v6b pilot's GO (EXPLORATORY, specified after the result; development data, no model pass).

The pilot re-ranks PreFLMR's top-20 with the similarity of the question's photograph to each candidate page's lead image.
E-VQA's photographs come from iNaturalist and from Google Landmarks v2, and Google Landmarks images are Wikimedia Commons
images: a page's lead image could be the question's photograph itself or a near copy, which a user's own photograph would
not be. This script asks where the gain comes from:
  - the gain in "gold page among the five read" by source of the photograph;
  - how similar the photograph is to the gold page's lead image (share above 0.9 and 0.95 cosine, DINOv2-base);
  - the gain when a near copy (cosine above 0.9) of the photograph is not allowed to help: for those candidates the image
    similarity is replaced by the question's mean similarity;
  - what the gain would be worth in always-retrieve accuracy if reading were as often right with and without the gold page
    among the five as it is now (from the main reader's development outputs; an extrapolation, not a measurement).

  python scripts/kbimage_rerank_diag.py --images <folder with thumbs.json + img/>   -> experiments/results/kbimage_rerank_diag.{json,md}
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.parse

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import kbimage_pilot as KP  # noqa: E402
from vgate import gate  # noqa: E402

N, K, W, DUP = 20, 5, 1.0, 0.9  # the pilot chose w = 1 on every fold


def embeddings(img, paths):
    """DINOv2 embeddings, cached next to the images (they are not part of the repository)."""
    cache = img / "emb_dinov2_top20.npz"
    E = dict(np.load(cache, allow_pickle=False)) if cache.exists() else {}
    todo = [p for p in paths if str(p) not in E]
    if todo:
        E.update({str(k): v for k, v in KP.embed("dinov2", todo).items()})
        np.savez(cache, **E)
    return E


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
    qimg = {f_: pathlib.Path(dp) / f_ for dp, _, fs in os.walk(ROOT / "data/raw/evqa_images") for f_ in fs}
    qids = sorted(recs)
    qpath = {i: qimg[pathlib.Path(recs[i]["image_file"]).name] for i in qids}
    E = embeddings(img, sorted(set(kb.values()) | set(qpath.values())))
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    rows = []
    for i in qids:
        L = list(raw.at[i, "retrieved_passage"])[:N]; gt = {KP.TITLE(x) for x in map(str, raw.at[i, "pos_item_ids"])}
        t = [KP.TITLE(x["passage_id"]) for x in L]; sc = np.array([x["score"] for x in L], float)
        qv = E[str(qpath[i])]
        sim = np.array([float(E[str(kb[x])] @ qv) if x in kb and str(kb[x]) in E else np.nan for x in t])
        fill = np.nanmean(sim) if np.isfinite(np.nanmean(sim)) else 0.0
        best = {}
        for j, x in enumerate(t):
            best.setdefault(x, sc[j])
        zs, gold = z(np.array([best[x] for x in t])), np.array([x in gt for x in t])
        s_all = np.where(np.isnan(sim), fill, sim)
        s_nodup = np.where(np.isnan(sim) | (sim > DUP), fill, sim)  # a near copy of the photograph may not help
        top = lambda s: np.lexsort((np.arange(N), -(zs + W * z(s))))[:K]
        gsim = float(np.nanmax(np.where(gold, sim, np.nan))) if gold.any() and np.isfinite(np.where(gold, sim, np.nan)).any() else np.nan
        rows.append(dict(qid=i, group=recs[i]["image_id"], source="landmarks" if "landmark" in recs[i]["image_file"] else "inaturalist",
                         base=bool(gold[:K].any()), rer=bool(gold[top(s_all)].any()), rer_nodup=bool(gold[top(s_nodup)].any()),
                         gold_in20=bool(gold.any()), gold_sim=gsim, any_dup=bool(np.nanmax(sim) > DUP) if np.isfinite(sim).any() else False))
    D = pd.DataFrame(rows).set_index("qid")

    def diff(mask, col):
        k = mask.to_numpy(); d = (D[col].to_numpy().astype(float) - D.base.to_numpy().astype(float))[k]; g = D.group.to_numpy()[k]
        pt, lo, hi = gate.cluster_bootstrap(g, lambda i: float(d[i].mean()), n_boot=a.n_boot, alpha=0.2)
        return dict(n=int(k.sum()), retriever=round(100 * float(D.base[k].mean()), 1), reranked=round(100 * float(D[col][k].mean()), 1),
                    diff=[round(100 * x, 1) for x in (pt, lo, hi)])
    rep = dict(n=len(D), weight=W, near_copy_threshold=DUP, by_source={}, near_copies={})
    every = pd.Series(True, index=D.index)
    for name, m in (("all", every), ("inaturalist", D.source == "inaturalist"), ("landmarks", D.source == "landmarks")):
        g = D[m & D.gold_sim.notna()]
        rep["by_source"][name] = dict(as_run=diff(m, "rer"), near_copies_not_used=diff(m, "rer_nodup"))
        rep["near_copies"][name] = dict(gold_page_with_image_in_top20=int(len(g)), median_similarity=round(float(g.gold_sim.median()), 3),
                                        share_above_0p9=round(100 * float((g.gold_sim > 0.9).mean()), 1), share_above_0p95=round(100 * float((g.gold_sim > 0.95).mean()), 1),
                                        questions_with_any_candidate_above_0p9=round(100 * float(D.any_dup[m].mean()), 1))
    # what the gain would be worth if reading stayed as often right with / without the gold page among the five (main reader, dev)
    fr = A.load(ROOT / "experiments/runs/J1", ROOT / "data", "evqa", "Qwen3-VL-4B-Instruct", "dev").reindex(D.index)
    acc_in, acc_out = float(fr.c_ev[D.base].mean()), float(fr.c_ev[~D.base].mean())
    rep["extrapolation"] = dict(accuracy_when_gold_page_among_five=round(100 * acc_in, 1), accuracy_when_not=round(100 * acc_out, 1),
                                always_retrieve_now=round(100 * float(fr.c_ev.mean()), 1),
                                implied_gain_as_run=round((acc_in - acc_out) * rep["by_source"]["all"]["as_run"]["diff"][0], 1),
                                implied_gain_near_copies_not_used=round((acc_in - acc_out) * rep["by_source"]["all"]["near_copies_not_used"]["diff"][0], 1))
    f = lambda v: f"{v['retriever']} -> {v['reranked']} ({v['diff'][0]:+.1f} [{v['diff'][1]:+.1f}, {v['diff'][2]:+.1f}], n = {v['n']})"
    L = ["# Where the gain of the top-20 image re-ranking comes from (diagnosis of PREREG v6b; exploratory; E-VQA dev; 80% intervals)", "",
         f"Gold page among the five read, retriever order -> re-ranked (w = {W}). \"Near copies not used\": a candidate image with cosine above {DUP} to the "
         f"question's photograph gets the question's mean similarity instead.", "",
         "| photographs | as run | near copies not used | gold lead image: median cosine | above 0.9 | above 0.95 |", "|---|---|---|---|---|---|"]
    for name, v in rep["by_source"].items():
        c = rep["near_copies"][name]
        L.append(f"| {name} | {f(v['as_run'])} | {f(v['near_copies_not_used'])} | {c['median_similarity']} | {c['share_above_0p9']}% | {c['share_above_0p95']}% |")
    e = rep["extrapolation"]
    L += ["", f"Extrapolation, not a measurement: the main reader is right on {e['accuracy_when_gold_page_among_five']}% of development questions when the gold page is among "
          f"the five read and on {e['accuracy_when_not']}% when it is not (always-retrieve: {e['always_retrieve_now']}%). If those two rates stayed as they are, the "
          f"re-ranking would be worth {e['implied_gain_as_run']:+.1f} points of always-retrieve accuracy as run and {e['implied_gain_near_copies_not_used']:+.1f} with near "
          f"copies not used. The questions that gain the gold page are harder than the average question that has it, so the reader pass can come out lower."]
    out = ROOT / "experiments/results/kbimage_rerank_diag"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

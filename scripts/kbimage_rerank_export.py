"""Write the re-ranked evidence of the PREREG v6b pilot for a reader pass (PREREG v6c): E-VQA development questions, each
with the five passages that the image re-ranking of PreFLMR's top-20 puts first (w = 1, the weight the pilot chose on
every fold), in the layout of data/processed/evqa.jsonl. Only `evidence` changes; passage texts come from the M2KR corpus.

  python scripts/kbimage_rerank_export.py --images <folder with thumbs.json + img/> [--no_near_copies]
      -> kaggle/_dataset_vgate-evqa-rerank/{processed/evqa.jsonl, splits/evqa_dev.txt, rerank_stats.json}

--no_near_copies writes the variant in which a candidate image with cosine above 0.9 to the question's photograph gets the
question's mean similarity (scripts/kbimage_rerank_diag.py), to processed_nocopy/evqa.jsonl.
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
import kbimage_pilot as KP  # noqa: E402
import kbimage_rerank_diag as RD  # noqa: E402

N, K, W, DUP = RD.N, RD.K, RD.W, RD.DUP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--no_near_copies", action="store_true")
    ap.add_argument("--split", default="dev", choices=["dev", "confirm"], help="confirm: the held-out split (PREREG v6d)")
    a = ap.parse_args()
    img = pathlib.Path(a.images)
    thumbs = json.loads((img / "thumbs.json").read_text())
    kb = {t: img / "img" / (urllib.parse.quote(t, safe="")[:150] + ".jpg") for t, u in thumbs.items() if u}
    kb = {t: p for t, p in kb.items() if p.exists()}
    dev = (ROOT / f"data/splits/evqa_{a.split}.txt").read_text().split()  # the questions of the chosen split
    raw = pd.read_parquet(ROOT / "data/raw/evqa_preflmr/data/test-00000-of-00001.parquet"); raw = raw[raw.question_id.isin(set(dev))].set_index("question_id")
    psg = pd.read_parquet(ROOT / "data/raw/m2kr/EVQA_passages/test_passages-00000-of-00001.parquet")
    psg = psg.drop_duplicates("passage_id", keep="first").set_index("passage_id")["passage_content"]  # as vgate.data.build_evqa
    recs = {r["qid"]: r for r in map(json.loads, open(ROOT / "data/processed/evqa.jsonl")) if r["qid"] in set(dev)}
    qimg = {f_: pathlib.Path(dp) / f_ for dp, _, fs in os.walk(ROOT / "data/raw/evqa_images") for f_ in fs}
    qpath = {i: qimg[pathlib.Path(recs[i]["image_file"]).name] for i in recs}
    E = RD.embeddings(img, sorted(set(kb.values()) | set(qpath.values())))
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    out, moved, gold5, cover, gold_page5, base_page5 = [], 0, 0, [], 0, 0
    for i in sorted(recs):
        L = list(raw.at[i, "retrieved_passage"])[:N]; gold = set(map(str, raw.at[i, "pos_item_ids"]))
        t = [KP.TITLE(x["passage_id"]) for x in L]; sc = np.array([x["score"] for x in L], float)
        qv = E[str(qpath[i])]
        sim = np.array([float(E[str(kb[x])] @ qv) if x in kb and str(kb[x]) in E else np.nan for x in t])
        cover.append(float(np.mean(~np.isnan(sim))))
        fill = np.nanmean(sim) if np.isfinite(np.nanmean(sim)) else 0.0
        sim = np.where(np.isnan(sim) | ((sim > DUP) if a.no_near_copies else False), fill, sim)
        best = {}
        for j, x in enumerate(t):
            best.setdefault(x, sc[j])
        order = np.lexsort((np.arange(N), -(z(np.array([best[x] for x in t])) + W * z(sim))))[:K]
        r = dict(recs[i])
        r["evidence"] = [dict(id=L[j]["passage_id"], title=t[j], score=float(L[j]["score"]), text=psg[L[j]["passage_id"]], retriever_rank=int(j) + 1) for j in order]
        r["lab_gold_in_top5"] = int(any(L[j]["passage_id"] in gold for j in order)); r["lab_gold_in_top1"] = int(L[order[0]]["passage_id"] in gold)
        moved += int(list(order) != list(range(K))); gold5 += r["lab_gold_in_top5"]
        gt = {KP.TITLE(x) for x in gold}
        gold_page5 += int(any(t[j] in gt for j in order)); base_page5 += int(any(x in gt for x in t[:K]))
        out.append(r)
    D = ROOT / ("kaggle/_dataset_vgate-evqa-rerank" + ("" if a.split == "dev" else "-confirm"))
    sub = "processed_nocopy" if a.no_near_copies else "processed"
    (D / sub).mkdir(parents=True, exist_ok=True); (D / "splits").mkdir(exist_ok=True)
    (D / sub / "evqa.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    (D / f"splits/evqa_{a.split}.txt").write_text("\n".join(dev) + "\n")
    stats = dict(split=a.split, n=len(out), evidence_changed=moved, gold_passage_among_five=round(100 * gold5 / len(out), 1), weight=W, near_copies_used=not a.no_near_copies,
                 candidates_with_image=round(100 * float(np.mean(cover)), 1), gold_page_among_five=dict(retriever=round(100 * base_page5 / len(out), 1), reranked=round(100 * gold_page5 / len(out), 1)))
    (D / f"rerank_stats{'_nocopy' if a.no_near_copies else ''}.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()

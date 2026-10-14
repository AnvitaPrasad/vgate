"""Build the unified question tables, frozen image-grouped splits, and the gate-feature whitelist.

One record per question (data/processed/<dataset>.jsonl):
  qid, dataset, image_id, image_file, question, answers, gold_answer, qtype, [infoseek: answer_eval,
  wikidata_value, wikidata_range, data_split], evidence (top-5 retrieved: id/title/score/text),
  retr_scores (top-20 retriever scores), oracle_text, and label-only fields prefixed `lab_`.

`lab_*` fields describe whether the retriever found the gold item. They are TARGETS for analysis and
the factorised P(hit) baseline, never gate features -- see FEATURE_BLACKLIST and tests/test_features.py.
"""
import hashlib
import json
import pathlib
import random
import re

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW, OUT, SPLITS = ROOT / "data/raw", ROOT / "data/processed", ROOT / "data/splits"
TOPK_TEXT, TOPK_SCORES, SEED, DEV_N = 5, 20, 2027, 1000
CHUNK_WORDS = 120  # InfoSeek pages are chunked; the best chunk per retrieved page is the evidence

# Anything that encodes retrieval success or the gold entity. Enforced by tests/test_features.py.
FEATURE_BLACKLIST = re.compile(r"^(lab_|Recall|Pseudo|MRR|pos_item|entity|wiki_title|oracle|answers|gold)")


def _chunks(text, n=CHUNK_WORDS):
    w = text.split()
    return [" ".join(w[i:i + n]) for i in range(0, max(len(w), 1), n)]


def build_evqa():
    ev = pd.read_parquet(RAW / "evqa_preflmr/data/test-00000-of-00001.parquet")
    m2 = pd.read_parquet(RAW / "m2kr/EVQA_data/test-00000-of-00001.parquet")[["question_id", "img_path"]]
    ev = ev.drop(columns=["img_path"]).merge(m2, on="question_id")
    psg = pd.read_parquet(RAW / "m2kr/EVQA_passages/test_passages-00000-of-00001.parquet")
    # 10 passage_ids occur twice with different text in the M2KR corpus; keep the first, deterministically.
    psg = psg.drop_duplicates("passage_id", keep="first").set_index("passage_id")["passage_content"]
    recs = []
    for r in ev.itertuples():
        top = list(r.retrieved_passage)
        gold = set(r.pos_item_ids)
        recs.append(dict(
            qid=r.question_id, dataset="evqa", image_id=r.image_id, image_file=r.img_path,
            question=r.question, answers=list(r.answers), gold_answer=r.gold_answer, qtype=r.question_type,
            evidence=[dict(id=x["passage_id"], title=x["passage_id"].removeprefix("WikiWeb_").rsplit("_", 1)[0],
                           score=float(x["score"]), text=psg[x["passage_id"]]) for x in top[:TOPK_TEXT]],
            retr_scores=[float(x["score"]) for x in top[:TOPK_SCORES]],
            oracle_text="\n".join(r.pos_item_contents),
            lab_gold_in_top1=int(top[0]["passage_id"] in gold),
            lab_gold_in_top5=int(any(x["passage_id"] in gold for x in top[:5])),
        ))
    return recs


def build_infoseek():
    from rank_bm25 import BM25Okapi
    inf = pd.read_parquet(RAW / "infoseek_evaclip/data/validation-00000-of-00001.parquet")
    m2 = pd.read_parquet(RAW / "m2kr/Infoseek_data/test-00000-of-00001.parquet")
    inf = inf.merge(m2[["question_id", "answer_eval", "wikidata_value", "wikidata_range", "question_type"]],
                    on="question_id", how="left")
    kb = json.load(open(RAW / "infoseek_kb/wiki_100_dict_v4.json"))
    tok = lambda s: re.findall(r"\w+", s.lower())
    recs = []
    for r in inf.itertuples():
        top = list(r.retrieved_passage)
        q = tok(r.question)
        evidence = []
        for x in top[:TOPK_TEXT]:
            page = kb[x["passage_id"]]
            chunks = [c for s in page["section_texts"] for c in _chunks(s) if c.strip()]
            best = chunks[int(np.argmax(BM25Okapi([tok(c) for c in chunks]).get_scores(q)))] if chunks else ""
            evidence.append(dict(id=x["passage_id"], title=x["title"], score=float(x["score"]), text=best))
        rng = r.wikidata_range
        recs.append(dict(
            qid=r.question_id, dataset="infoseek", image_id=r.image_id, image_file=r.img_path,
            question=r.question, answers=list(r.answers), gold_answer=r.gold_answer, qtype=r.question_type,
            answer_eval=[a if isinstance(a, str) else json.loads(json.dumps(a, default=float)) for a in r.answer_eval],
            wikidata_value=None if pd.isna(r.wikidata_value) else float(r.wikidata_value),
            wikidata_range=None if rng is None else [float(v) for v in rng],
            data_split=r.data_split,
            evidence=evidence,
            retr_scores=[float(x["score"]) for x in top[:TOPK_SCORES]],
            oracle_text="\n".join(r.pos_item_contents),
            lab_gold_in_top1=int(top[0]["passage_id"] in set(r.pos_item_ids)),
            lab_gold_in_top5=int(any(x["passage_id"] in set(r.pos_item_ids) for x in top[:5])),
        ))
    return recs


def build_okvqa():
    """OK-VQA (PREREG v2): M2KR test questions + dense bge retrieval (scripts/build_okvqa_retrieval.py)."""
    q = pd.read_parquet(RAW / "m2kr/OKVQA_data/test-00000-of-00001.parquet")
    p = pd.read_parquet(RAW / "m2kr/OKVQA_passages/test_passages-00000-of-00001.parquet").drop_duplicates("passage_id")
    p = p.set_index("passage_id")
    ret = pd.read_parquet(RAW / "okvqa_retrieval.parquet").set_index("question_id")["retrieved"]

    def title_text(pid):
        c = p.passage_content[pid]
        m = re.match(r'title:\s*"?(.*?)"?\s+content:\s*(.*)', c, re.S)
        return (m.group(1), m.group(2)) if m else (p.title.get(pid, ""), c)

    recs = []
    for r in q.itertuples():
        qid = str(r.question_id)
        top = list(ret[qid])
        gold = set(map(str, r.pos_item_ids))
        ev = []
        for x in top[:TOPK_TEXT]:
            t, txt = title_text(x["passage_id"])
            ev.append(dict(id=x["passage_id"], title=t, score=float(x["score"]), text=txt))
        recs.append(dict(
            qid=qid, dataset="okvqa", image_id=str(r.img_key), image_file=r.img_file_name,
            question=r.question, answers=list(r.answers), gold_answer=r.gold_answer, qtype="okvqa",
            evidence=ev, retr_scores=[float(x["score"]) for x in top[:TOPK_SCORES]],
            oracle_text="\n".join(list(r.pos_item_contents)[:1]),
            lab_gold_in_top1=int(top[0]["passage_id"] in gold),
            lab_gold_in_top5=int(any(x["passage_id"] in gold for x in top[:5])),
        ))
    return recs


def grouped_split(recs, n_dev=DEV_N, seed=SEED):
    """Dev = random image groups until >= n_dev questions; the rest is the confirmatory set."""
    images = sorted({r["image_id"] for r in recs})
    random.Random(seed).shuffle(images)
    by_img = {}
    for r in recs:
        by_img.setdefault(r["image_id"], []).append(r["qid"])
    dev = []
    for im in images:
        if len(dev) >= n_dev:
            break
        dev += by_img[im]
    dev_set = set(dev)
    return sorted(dev), sorted(r["qid"] for r in recs if r["qid"] not in dev_set)


def main(only=None):
    OUT.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    digests = {}
    builders = [("evqa", build_evqa), ("infoseek", build_infoseek), ("okvqa", build_okvqa)]
    for name, build in [b for b in builders if only is None or b[0] in only]:
        recs = build()
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        for split, qids in zip(["dev", "confirm"], grouped_split(recs)):
            p = SPLITS / f"{name}_{split}.txt"
            p.write_text("\n".join(qids) + "\n")
            digests[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
            print(f"{name}/{split}: {len(qids)} questions")
    old = dict(l.split()[::-1] for l in (SPLITS / "SPLITS.sha256").read_text().splitlines()) if (SPLITS / "SPLITS.sha256").exists() else {}
    old.update(digests)  # never rewrites the hashes of datasets not rebuilt
    (SPLITS / "SPLITS.sha256").write_text("".join(f"{v}  {k}\n" for k, v in sorted(old.items())))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)

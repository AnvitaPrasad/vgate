"""System check of the live pipeline (PREREG v2f): run it end to end, one query at a time, on held-out samples.

  python -m vgate.live_eval --index /path/index --roots evqa=/data,infoseek=/data,okvqa=/okvqa \
      --n okvqa=400,evqa=200,infoseek=200 --out runs/J9 [--model Qwen/Qwen3-VL-4B-Instruct]
Every pass is executed for every query (run_all), so never-, always-retrieve and VGate can all be scored afterwards.
Writes <out>/live__<dataset>.jsonl (one row per query, incl. per-stage wall time) and live_hidden__<dataset>.npy.
`python -m vgate.live_eval --build_index ...` builds the passage matrix the retriever needs.
"""
import argparse
import json
import os
import pathlib
import random
import re
import time

import numpy as np

from .live import GATES_DIR, RETRIEVER, Gates, LivePipeline, Retriever
from .models import VLM, load_image
from .run import resolve_images


def build_index(passages_parquet, out_dir, device="cuda:0"):
    """Embed the M2KR OK-VQA passages exactly as scripts/build_okvqa_retrieval.py did; store float16 rows."""
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(passages_parquet).drop_duplicates("passage_id")
    m = SentenceTransformer(RETRIEVER, device=device)
    m.max_seq_length = 256
    t0 = time.time()
    P = m.encode(p.passage_content.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False,
                 convert_to_numpy=True)
    np.save(out / "passage_emb.npy", P.astype(np.float16))
    with open(out / "passages.jsonl", "w") as f:
        for pid, c, title in zip(p.passage_id, p.passage_content, p.title if "title" in p else [""] * len(p)):
            mt = re.match(r'title:\s*"?(.*?)"?\s+content:\s*(.*)', c, re.S)  # same parsing as data.build_okvqa
            t, txt = (mt.group(1), mt.group(2)) if mt else (title, c)
            f.write(json.dumps(dict(id=str(pid), title=t, text=txt)) + "\n")
    print(f"index: {P.shape} in {time.time() - t0:.0f}s -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_index", default="", help="passages parquet; build the index into --index and exit")
    ap.add_argument("--index", required=True)
    ap.add_argument("--roots", default="", help="dataset=data_root,... (each root has processed/, splits/, images/)")
    ap.add_argument("--n", default="okvqa=400,evqa=200,infoseek=200")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out", default="runs/J9")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--index_device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=2027)
    a = ap.parse_args()
    if a.build_index:
        return build_index(a.build_index, a.index, a.index_device)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mname = a.model.split("/")[-1]
    pipe = LivePipeline(VLM(a.model, dtype=a.dtype, device=a.device), Gates(GATES_DIR / f"{mname}.npz"),
                        Retriever(a.index, device="cpu"))  # query encoder on CPU, as on a GPU-on-demand host
    roots, n = dict(x.split("=") for x in a.roots.split(",")), {k: int(v) for k, v in (x.split("=") for x in a.n.split(","))}
    for d, root in roots.items():
        recs = {r["qid"]: r for r in map(json.loads, open(pathlib.Path(root) / "processed" / f"{d}.jsonl"))}
        qids = sorted((pathlib.Path(root) / "splits" / f"{d}_confirm.txt").read_text().split())
        qids = random.Random(a.seed).sample(qids, min(n.get(d, 0), len(qids)))
        img = resolve_images(os.path.join(root, "images"))
        hid, t0 = [], time.time()
        with open(out / f"live__{d}.jsonl", "w") as f:
            for i, q in enumerate(qids):
                r = recs[q]
                res = pipe.answer(load_image(img[os.path.basename(r["image_file"])]), r["question"], run_all=True)
                hid.append(res.pop("hidden"))
                res["evidence"] = [dict(id=e["id"], title=e["title"], score=e["score"]) for e in res["evidence"]]
                f.write(json.dumps(dict(qid=q, dataset=d, **res)) + "\n")
                f.flush()
                if i % 50 == 0:
                    print(f"[{d}] {i + 1}/{len(qids)}  {(time.time() - t0) / (i + 1):.2f}s/q", flush=True)
        np.save(out / f"live_hidden__{d}.npy", np.stack(hid))
        (out / f"live_qids__{d}.json").write_text(json.dumps(qids))


if __name__ == "__main__":
    main()

"""CVQA (Romero et al., NeurIPS 2024 Datasets and Benchmarks; HF `afaji/cvqa`) as a fourth benchmark (PREREG v7):
four-option multiple choice on culturally diverse images, English question and options. Evidence comes from the
multilingual Wikipedia corpus of M4-RAG (Anugraha et al., arXiv 2512.05959; HF `davidanugraha/M4-RAG`, config cvqa).
Everything here runs inside the GPU job; the images are written to a temporary folder and are not stored by us.

  python -m vgate.cvqa questions --raw <cvqa parquet dir> --images <dir> --data <data dir>
      -> <data>/processed/cvqa.jsonl (no evidence yet) and <data>/splits/cvqa_{dev,confirm}.txt
  python -m vgate.cvqa retrieve --data <data dir> --ent <run dir>[,<run dir>] --corpus <M4-RAG cvqa parquet dir>
      -> rewrites <data>/processed/cvqa.jsonl with evidence (top-5 pages, the best 120-word chunk of each) and
         retr_scores (top-20 page scores)

Retrieval: dense, bge-m3 (multilingual), query = the reader's entity name for the image + the question (the options
are not part of the query); a page is embedded as its title and first 150 words.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from . import data

EMB = os.environ.get("VGATE_EMB", "BAAI/bge-m3")  # the environment variable is for tests only
DEV_N, LEAD_WORDS, CHUNK_WORDS, MAX_CHUNKS, MAX_ENT_WORDS = 2000, 150, 120, 20, 12
MAX_TOKENS = 220  # cap of one evidence passage in the retriever's tokens (120 English words are about 170)
MODEL = "Qwen3-VL-4B-Instruct"  # the reader whose entity names form the queries


def questions(raw, images):
    """Unified records (no evidence). image_id is a hash of the image bytes, so questions on one image share a group."""
    import pyarrow.parquet as pq
    from PIL import Image, ImageOps
    images = pathlib.Path(images); images.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None  # a few photographs are very large
    recs, skipped = [], 0
    for f in sorted(glob.glob(str(pathlib.Path(raw) / "**/*.parquet"), recursive=True)):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=32):
            for r in batch.to_pylist():
                opts = [str(o).strip() for o in (r["Translated Options"] or [])]
                if len(opts) != 4 or r["Label"] not in (0, 1, 2, 3) or not r["image"] or not r["image"].get("bytes"):
                    skipped += 1
                    continue
                b = r["image"]["bytes"]
                h = hashlib.md5(b).hexdigest()[:16]
                if not (images / f"{h}.jpg").exists():
                    try:
                        im = ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert("RGB")
                        im.thumbnail((1024, 1024))
                        im.save(images / f"{h}.jpg", quality=90)
                    except Exception as e:  # unreadable image: the question is dropped and counted
                        print("image error", r["ID"], repr(e)[:80], flush=True)
                        skipped += 1
                        continue
                recs.append(dict(qid=str(r["ID"]), dataset="cvqa", image_id=h, image_file=f"{h}.jpg",
                                 question=str(r["Translated Question"]).strip(), options=opts,
                                 answers=["ABCD"[r["Label"]]], gold_answer=opts[r["Label"]], qtype=r["Category"],
                                 subset=r["Subset"], image_type=r["Image Type"], evidence=[], retr_scores=[],
                                 oracle_text="", lab_gold_in_top1=None, lab_gold_in_top5=None))
    assert len({r["qid"] for r in recs}) == len(recs), "question ids are not unique"
    print(f"cvqa: {len(recs)} questions, {len({r['image_id'] for r in recs})} images, {skipped} skipped", flush=True)
    return sorted(recs, key=lambda r: r["qid"])


def page_parts(content):
    """M4-RAG stores a page as {"heading": [...], "content": [[paragraph, ...], ...]} -> list of paragraphs."""
    if isinstance(content, str):
        return [content]
    secs = (content or {}).get("content") or []
    return [p.strip() for s in secs for p in (s if isinstance(s, (list, tuple)) else [s]) if isinstance(p, str) and p.strip()]


def units(text):
    """Words; for scripts written without spaces (mean 'word' over 15 characters) two characters count as one word."""
    w = text.split()
    if w and len(text) / len(w) > 15:
        t = " ".join(w)
        return [t[i:i + 2] for i in range(0, len(t), 2)], ""
    return w, " "


def chunks_of(text, n=CHUNK_WORDS):
    u, sep = units(text)
    return [sep.join(u[i:i + n]) for i in range(0, len(u), n)]


def embed_local(texts, dev, bs=96, max_len=256):
    """Dense embeddings of EMB (CLS token, L2-normalised) on one device; fp16 on a GPU. Rows in the order of `texts`."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB)
    m = AutoModel.from_pretrained(EMB, dtype=torch.float16 if str(dev).startswith("cuda") else torch.float32).to(dev).eval()
    order = np.argsort([-len(t) for t in texts])  # long first: similar lengths share a batch
    out = [None] * len(texts)
    for i in range(0, len(order), bs):
        j = order[i:i + bs]
        x = tok([texts[k] for k in j], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        with torch.inference_mode():
            v = torch.nn.functional.normalize(m(**x).last_hidden_state[:, 0].float(), dim=-1).cpu().numpy()
        for k, row in zip(j, v):
            out[k] = row.astype(np.float16)
    return np.stack(out) if texts else np.zeros((0, 1), np.float16)


def embed(texts, devices):
    """Embeddings of `texts` (float32, rows in order). On GPUs: one worker process per device, each seeing only its own
    GPU through CUDA_VISIBLE_DEVICES (two threads sharing a process failed on Kaggle with a CUDA driver error)."""
    if not texts:
        return np.zeros((0, 1), np.float32)
    if not str(devices[0]).startswith("cuda"):
        return embed_local(texts, devices[0]).astype(np.float32)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="emb_"))
    parts, procs = [np.arange(len(texts))[i::len(devices)] for i in range(len(devices))], []
    for i, (d, idx) in enumerate(zip(devices, parts)):
        (tmp / f"t{i}.json").write_text(json.dumps([texts[k] for k in idx], ensure_ascii=False))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(d).split(":")[-1]}
        procs.append(subprocess.Popen([sys.executable, "-m", "vgate.cvqa", "embed", "--data", str(tmp), "--texts", str(tmp / f"t{i}.json"),
                                       "--out", str(tmp / f"e{i}.npy")], env=env))
    while any(q.poll() is None for q in procs):
        if any(q.poll() not in (None, 0) for q in procs):  # one worker failed: stop the others at once
            [q.kill() for q in procs if q.poll() is None]
        time.sleep(2)
    assert all(q.returncode == 0 for q in procs), f"embedding worker failed: {[q.returncode for q in procs]}"
    first = np.load(tmp / "e0.npy")
    out = np.zeros((len(texts), first.shape[1]), np.float32)
    for i, idx in enumerate(parts):
        out[idx] = first if i == 0 else np.load(tmp / f"e{i}.npy")
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def top_pages(Q, D, k=data.TOPK_SCORES, bs=256):
    """Indices and cosine scores of the k best rows of D for every row of Q (exact search, on the CPU)."""
    k = min(k, len(D))
    idx, sc = [], []
    for i in range(0, len(Q), bs):
        S = Q[i:i + bs] @ D.T
        j = np.argpartition(-S, k - 1, axis=1)[:, :k]
        v = np.take_along_axis(S, j, axis=1)
        o = np.argsort(-v, axis=1)
        idx.append(np.take_along_axis(j, o, axis=1)); sc.append(np.take_along_axis(v, o, axis=1))
    return np.concatenate(idx), np.concatenate(sc)


def retrieve(recs, ent, corpus, devices):
    """Adds evidence and retr_scores to `recs` in place. `ent`: {qid: entity name given by the reader}."""
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(pathlib.Path(corpus) / "**/*.parquet"), recursive=True))
    ids, titles, langs, leads = [], [], [], []
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["id", "title", "url", "content"]):
            for r in batch.to_pylist():
                ids.append(r["id"]); titles.append(r["title"] or "")
                langs.append((r["url"] or "").split("//")[-1].split(".")[0])
                u, sep = units(" ".join(page_parts(r["content"])))
                leads.append(sep.join(u[:LEAD_WORDS]))
    print(f"corpus: {len(ids)} pages", flush=True)
    D = embed([f"{t}\n{l}" for t, l in zip(titles, leads)], devices)
    queries = [f"{' '.join(ent.get(r['qid'], '').split()[:MAX_ENT_WORDS])} {r['question']}".strip() for r in recs]
    Q = embed(queries, devices)
    idx, sc = top_pages(Q, D)
    del D
    need = {ids[j] for row in idx for j in row[:data.TOPK_TEXT]}
    chunks, owner = [], []  # the first MAX_CHUNKS chunks of every page some question reads
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=256, columns=["id", "content"]):
            for r in batch.to_pylist():
                if r["id"] in need:
                    cs = [c for c in chunks_of(" ".join(page_parts(r["content"]))) if c.strip()][:MAX_CHUNKS]
                    chunks += cs or [""]; owner += [r["id"]] * len(cs or [""])
    print(f"pages read by some question: {len(need)}, chunks: {len(chunks)}", flush=True)
    C = embed(chunks, devices)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB)

    def capped(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return text if len(ids) <= MAX_TOKENS else tok.decode(ids[:MAX_TOKENS]) + " ..."
    rows = {}
    for n, pid in enumerate(owner):
        rows.setdefault(pid, []).append(n)
    for i, r in enumerate(recs):
        r["retr_scores"] = [round(float(x), 5) for x in sc[i]]
        r["evidence"] = []
        for j, s in zip(idx[i][:data.TOPK_TEXT], sc[i][:data.TOPK_TEXT]):
            cand = rows[ids[j]]
            best = cand[int(np.argmax(C[cand] @ Q[i]))]
            r["evidence"].append(dict(id=ids[j], title=titles[j], score=round(float(s), 5), text=capped(chunks[best]), lang=langs[j]))
        r["query"] = queries[i]
    return recs


def write(recs, out, splits=True):
    out = pathlib.Path(out)
    (out / "processed").mkdir(parents=True, exist_ok=True); (out / "splits").mkdir(parents=True, exist_ok=True)
    with open(out / "processed" / "cvqa.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if splits:
        dev, confirm = data.grouped_split(recs, n_dev=DEV_N)
        (out / "splits" / "cvqa_dev.txt").write_text("\n".join(dev) + "\n")
        (out / "splits" / "cvqa_confirm.txt").write_text("\n".join(confirm) + "\n")
        print(f"splits: dev {len(dev)}, confirm {len(confirm)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["questions", "retrieve", "embed", "selftest"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--raw"); ap.add_argument("--images"); ap.add_argument("--ent"); ap.add_argument("--corpus")
    ap.add_argument("--devices", default="cuda:0")
    ap.add_argument("--texts"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.step == "embed":  # worker of embed(): sees one GPU as cuda:0
        np.save(a.out, embed_local(json.loads(pathlib.Path(a.texts).read_text()), os.environ.get("VGATE_EMBED_DEVICE", "cuda:0")))
        return
    if a.step == "selftest":  # a few texts through every device, before anything long runs
        e = embed([f"test sentence number {i}" for i in range(40)], a.devices.split(","))
        assert e.shape[0] == 40 and np.isfinite(e).all() and abs(float(np.linalg.norm(e[0])) - 1) < 1e-2, "embedding self-test failed"
        print("embedding self-test ok", e.shape, flush=True)
        return
    if a.step == "questions":
        write(questions(a.raw, a.images), a.data)
    else:
        recs = [json.loads(l) for l in open(pathlib.Path(a.data) / "processed" / "cvqa.jsonl")]
        ent = {}
        for d in a.ent.split(","):
            p = pathlib.Path(d) / f"cvqa__{MODEL}__ent.jsonl"
            ent.update({x["qid"]: x["text"] for x in map(json.loads, open(p))})
        missing = sum(r["qid"] not in ent for r in recs)
        print(f"entity names: {len(ent)}, questions without one: {missing}", flush=True)
        write(retrieve(recs, ent, a.corpus, a.devices.split(",")), a.data, splits=False)


if __name__ == "__main__":
    main()

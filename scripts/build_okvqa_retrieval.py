"""OK-VQA retrieval (PREREG v2): dense bge-base-en-v1.5 over the 114,809 M2KR OK-VQA passages.
Query = dataset-provided image caption + question (no VLM pass). Writes data/raw/okvqa_retrieval.parquet with the
top-20 (passage_id, score) per question_id."""
import pathlib

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/m2kr"
q = pd.read_parquet(RAW / "OKVQA_data/test-00000-of-00001.parquet")
p = pd.read_parquet(RAW / "OKVQA_passages/test_passages-00000-of-00001.parquet").drop_duplicates("passage_id")
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev)
m.max_seq_length = 256  # passages are ~150 tokens; avoids padding every batch to 512
P = m.encode(p.passage_content.tolist(), batch_size=32, normalize_embeddings=True, show_progress_bar=True,
             convert_to_numpy=True).astype(np.float32)
queries = [f"{c['caption'] if isinstance(c, dict) else ''} {qq}".strip() for c, qq in zip(q.img_caption, q.question)]
Q = m.encode(["Represent this sentence for searching relevant passages: " + s for s in queries], batch_size=128,
             normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
S = Q @ P.T
top = np.argsort(-S, axis=1)[:, :20]
pid = p.passage_id.to_numpy()
out = pd.DataFrame({"question_id": q.question_id.astype(str),
                    "retrieved": [[{"passage_id": str(pid[j]), "score": float(S[i, j])} for j in top[i]] for i in range(len(q))]})
out.to_parquet(ROOT / "data/raw/okvqa_retrieval.parquet")
print("done", out.shape, "device", dev)

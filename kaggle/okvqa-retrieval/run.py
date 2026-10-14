"""Kaggle copy of scripts/build_okvqa_retrieval.py (same model, query and top-k; only paths and batch size differ)."""
import subprocess
subprocess.run("pip install -q sentence-transformers 2>&1 | tail -1", shell=True)

import pathlib

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

ROOT = pathlib.Path("/kaggle/working")
RAW = ROOT / "data/raw/m2kr"
from huggingface_hub import hf_hub_download
for _f in ["OKVQA_data/test-00000-of-00001.parquet", "OKVQA_passages/test_passages-00000-of-00001.parquet"]:
    hf_hub_download("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR", _f, repo_type="dataset", local_dir=RAW)
(ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
q = pd.read_parquet(RAW / "OKVQA_data/test-00000-of-00001.parquet")
p = pd.read_parquet(RAW / "OKVQA_passages/test_passages-00000-of-00001.parquet").drop_duplicates("passage_id")
dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
m = SentenceTransformer("BAAI/bge-base-en-v1.5", device=dev)
m.max_seq_length = 256  # passages are ~150 tokens; avoids padding every batch to 512
P = m.encode(p.passage_content.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False,
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

import shutil; shutil.copy(ROOT / "data/raw/okvqa_retrieval.parquet", "/kaggle/working/okvqa_retrieval.parquet"); shutil.rmtree(ROOT / "data")

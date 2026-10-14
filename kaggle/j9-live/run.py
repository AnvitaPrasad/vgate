"""J9 (PREREG v2f): system check of the live pipeline. Builds the passage index (bge-base over the M2KR OK-VQA
passages), then runs vgate.live_eval one query at a time on held-out samples of the three datasets, executing every
pass for every query. Outputs: /kaggle/working/J9/{index/, live__*.jsonl, live_hidden__*.npy}."""
import glob, json, os, subprocess, sys, time

def sh(c):
    t = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(f"$ {c[:220]} [{time.time()-t:.0f}s rc={r.returncode}]\n{r.stdout[-3500:]}{r.stderr[-3000:]}", flush=True)
    return r.returncode

# Kaggle flattens the single-folder code zip into /kaggle/input/datasets/<user>/vgate-code/: rebuild the package.
SRC = os.path.dirname((glob.glob("/kaggle/input/**/vgate/run.py", recursive=True)
                       or glob.glob("/kaggle/input/**/vgate-code/run.py", recursive=True))[0])
PKG = "/kaggle/working/pkg"; os.makedirs(PKG, exist_ok=True)
subprocess.run(f"rm -rf {PKG}/vgate && cp -r {SRC} {PKG}/vgate", shell=True, check=True)
CODE = f"{PKG}/vgate"
ROOTS = {d: os.path.dirname(os.path.dirname(glob.glob(f"/kaggle/input/**/processed/{d}.jsonl", recursive=True)[0]))
         for d in ("evqa", "infoseek", "okvqa")}
sys.path.insert(0, os.path.dirname(CODE))
from vgate._build import BUILD
EXPECTED = "63bce7ece0a4"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, "roots:", ROOTS, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh(f"ls -la {CODE} {CODE}/gates; nvidia-smi --query-gpu=name,memory.total --format=csv")
sh('pip install -q "transformers==5.16.1" accelerate sentence-transformers 2>&1 | tail -2; python -c "import transformers, sentence_transformers as st;print(transformers.__version__, st.__version__)"')

OUT = "/kaggle/working/J9"; os.makedirs(OUT, exist_ok=True)
ENV = f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/tmp/hf"
from huggingface_hub import hf_hub_download
PQ = hf_hub_download("BByrneLab/multi_task_multi_modal_knowledge_retrieval_benchmark_M2KR",
                     "OKVQA_passages/test_passages-00000-of-00001.parquet", repo_type="dataset", local_dir="/tmp/m2kr")
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
rc = sh(f"{ENV} CUDA_VISIBLE_DEVICES={1 if NGPU > 1 else 0} python -m vgate.live_eval --build_index {PQ} --index {OUT}/index > {OUT}/log_index.txt 2>&1; tail -3 {OUT}/log_index.txt")
if rc == 0:
    roots = ",".join(f"{d}={r}" for d, r in ROOTS.items())
    sh(f"{ENV} CUDA_VISIBLE_DEVICES=0 python -m vgate.live_eval --index {OUT}/index --roots {roots} "
       f"--n okvqa=400,evqa=200,infoseek=200 --out {OUT} > {OUT}/log_live.txt 2>&1; "
       f"grep -E 'Error|Traceback|s/q' -A4 {OUT}/log_live.txt | tail -30")
for f in sorted(glob.glob(OUT + "/live__*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    med = lambda k: sorted(r["seconds"].get(k, 0) for r in rows)[len(rows) // 2]
    print(f"{os.path.basename(f)}: n={len(rows)} search={sum(r['search'] for r in rows)} read={sum(r['read'] for r in rows)} "
          f"median s: closed_book {med('closed_book'):.2f} caption {med('caption'):.2f} retrieval {med('retrieval'):.2f} "
          f"evidence {med('evidence'):.2f}", flush=True)
sh(f"ls -la {OUT} {OUT}/index; rm -rf {PKG}")

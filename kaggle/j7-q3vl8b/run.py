"""J7 model-4 replication (PREREG v2c): Qwen3-VL-8B-Instruct, ne + ev on E-VQA and InfoSeek (dev, then held-out).
The fp16 weights (~16 GiB) do not fit one T4, so ONE process holds the model split across both GPUs (--device auto).
Dev splits run first so that a time-out still leaves usable development data. Resumable per qid."""
import glob, json, os, subprocess, sys, time

def sh(c):
    t = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(f"$ {c[:200]} [{time.time()-t:.0f}s rc={r.returncode}]\n{r.stdout[-3000:]}{r.stderr[-2500:]}", flush=True)
    return r.returncode

# Kaggle flattens the single-folder code zip into /kaggle/input/datasets/<user>/vgate-code/: rebuild the package.
SRC = os.path.dirname((glob.glob("/kaggle/input/**/vgate/run.py", recursive=True)
                       or glob.glob("/kaggle/input/**/vgate-code/run.py", recursive=True))[0])
PKG = "/kaggle/working/pkg"; os.makedirs(PKG, exist_ok=True)
subprocess.run(f"rm -rf {PKG}/vgate && cp -r {SRC} {PKG}/vgate", shell=True, check=True)
CODE = f"{PKG}/vgate"
DATA = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/splits/SPLITS.sha256", recursive=True)[0]))  # parent of splits/ and processed/
IMGS = os.path.join(DATA, "images")
sys.path.insert(0, os.path.dirname(CODE))
from vgate._build import BUILD
EXPECTED = "fc206451eba7"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh("sha256sum " + " ".join(sorted(glob.glob(f"{DATA}/splits/*.txt"))) + f"; cat {DATA}/splits/SPLITS.sha256")
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; python -c 'import torch;print(torch.__version__, torch.version.cuda)'")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
if NGPU < 2:
    print("BLOCKED: Qwen3-VL-8B fp16 needs two T4s; this session has", NGPU, flush=True)
    sys.exit(0)

OUT = "/kaggle/working/J7"; os.makedirs(OUT, exist_ok=True)
M = "Qwen/Qwen3-VL-8B-Instruct"
ENV = f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/tmp/hf"
BASE = (f"python -m vgate.run --data {DATA} --images {IMGS} --out {OUT} --dtype float16 --attn sdpa --device auto --model {M} --conditions ne,ev")
T0, BUDGET = time.time(), 10.5 * 3600  # Kaggle kills the session at 12 h; stop early so that outputs are saved
ok = True
for i, (split, d) in enumerate([(s, d) for s in ("dev", "confirm") for d in ("evqa", "infoseek")]):
    for attempt, batch in enumerate([8, 4, 2]):
        left = int(BUDGET - (time.time() - T0))
        if left < 300:
            print(f"TIME BUDGET EXHAUSTED before {d}/{split}", flush=True); ok = False; break
        log = f"{OUT}/log_{i}_{d}_{split}_try{attempt}.txt"
        rc = sh(f"{ENV} timeout {left} {BASE} --dataset {d} --split {split} --batch {batch} > {log} 2>&1")
        sh(f"grep -E 'Error|Traceback|s/q|questions' -A3 {log} | tail -12; nvidia-smi --query-gpu=memory.used --format=csv,noheader")
        if rc == 0:
            break
    else:
        ok = False
    if not ok:
        break  # a crash on the first split would repeat on the others: fail fast and show the log
print("ALL SPLITS DONE" if ok else "INCOMPLETE (see logs above); outputs so far are kept", flush=True)

import math
for f in sorted(glob.glob(OUT + "/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    nan = sum(any(math.isnan(x) for x in r.get("tok_logprobs", [])) for r in rows)
    empty = sum((r.get("text", "x") or "").strip() == "" for r in rows)
    gs = sum(r.get("gpu_s", 0) for r in rows) / max(len(rows), 1)
    npr = sum(r.get("n_prompt", 0) for r in rows) / max(len(rows), 1)
    ngen = sorted(r.get("n_gen", 0) for r in rows)
    trunc = sum(r.get("truncated", False) for r in rows)
    print(f"{os.path.basename(f)}: n={len(rows)} nan={nan} empty={empty} gpu_s/q={gs:.2f} prompt_tok={npr:.0f} "
          f"gen_tok median={ngen[len(ngen)//2] if ngen else 0} max={ngen[-1] if ngen else 0} truncated={trunc}", flush=True)
sh(f"ls -la {OUT} | head -40; rm -rf {PKG}")

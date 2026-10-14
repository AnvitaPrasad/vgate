"""J23 (PREREG v13): a larger reader, Qwen3-VL-32B-Instruct with 4-bit weights, one copy over two T4s. Closed-book and top-5
evidence passes on the first N held-out questions of a frozen seeded order per benchmark. LIMIT > 0: smoke run (timing only;
no accuracy is printed in either mode)."""
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
ROOTS = {d: os.path.dirname(os.path.dirname(glob.glob(f"/kaggle/input/**/processed/{d}.jsonl", recursive=True)[0]))
         for d in ("evqa", "infoseek", "okvqa")}
sys.path.insert(0, os.path.dirname(CODE))
from vgate._build import BUILD
EXPECTED = "1ce75901f019"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, "roots:", ROOTS, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh('pip install -q "transformers==5.16.1" accelerate bitsandbytes 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
if NGPU < 2:
    print("BLOCKED: the 32B reader needs two T4s; this session has", NGPU, flush=True)
    sys.exit(0)

LIMIT = 0    # questions per benchmark for a smoke run (12 on 2026-10-11); 0 = the full run on the first N of the frozen order
N = 500      # by the rule of PREREG v13: 11.3 s per question for both passes in the smoke run
OUT = "/kaggle/working/J23"; os.makedirs(OUT, exist_ok=True)
M = "Qwen/Qwen3-VL-32B-Instruct"
ENV = f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/tmp/hf"
HASH = dict(evqa="8c62d70adabfed5f70c2937fb7e8f79ad6ca2bbce155a7c0ccaf5aa4fd597946", infoseek="ff3d41a4ba991f5e25f8916201052edf803e6a72490316528971c96c712647a6",
            okvqa="03fcf4e8ef55d4768b0dcd3fc7e3fd7be8c33ab8e97f31c6c5722ba04c3aa4a5")  # data/splits/<benchmark>_big.txt (first 1,000 of the order)
import hashlib, shutil
import numpy as np
DATA = {}
for d in ("evqa", "infoseek", "okvqa"):  # a writable copy of the records and splits, plus the frozen order
    DATA[d] = f"/kaggle/working/data_{d}"; os.makedirs(f"{DATA[d]}/processed", exist_ok=True); os.makedirs(f"{DATA[d]}/splits", exist_ok=True)
    shutil.copy(f"{ROOTS[d]}/processed/{d}.jsonl", f"{DATA[d]}/processed/{d}.jsonl")
    q = sorted(open(glob.glob(f"/kaggle/input/**/splits/{d}_confirm.txt", recursive=True)[0]).read().split())
    order = [q[i] for i in np.random.RandomState(2027).permutation(len(q))][:1000]
    text = "\n".join(order) + "\n"
    assert hashlib.sha256(text.encode()).hexdigest() == HASH[d], f"the frozen order of {d} does not match"
    open(f"{DATA[d]}/splits/{d}_big.txt", "w").write("\n".join(order[:N]) + "\n")
print("frozen orders verified; N =", N, "LIMIT =", LIMIT, flush=True)
sh(f"{ENV} python -c \"from huggingface_hub import snapshot_download; print(snapshot_download('{M}'))\" 2>&1 | tail -1; df -h /tmp | tail -1")
T0, BUDGET = time.time(), 10.5 * 3600  # stop before the platform limit so that outputs are saved
ok = True
for i, d in enumerate(("evqa", "infoseek", "okvqa")):
    left = int(BUDGET - (time.time() - T0))
    if left < 600:
        print(f"TIME BUDGET EXHAUSTED before {d}", flush=True); ok = False; break
    # Version 2 ran out of memory on the second card at E-VQA evidence question 292 (weights 13 GiB there, 1.5 GiB left for a long
    # prompt). Now: less of the weights on the second card, and the runner skips (and prints) a question that still does not fit.
    for attempt, mm in enumerate(("10GiB,11500MiB", "10500MiB,12GiB", "10GiB,13GiB")):  # the run resumes: finished questions are kept
        left = int(BUDGET - (time.time() - T0))
        log = f"{OUT}/log_{i}_{d}_try{attempt}.txt"
        rc = sh(f"{ENV} VGATE_MAX_MEMORY={mm} timeout {left} python -m vgate.run --data {DATA[d]} --images {ROOTS[d]}/images --out {OUT} --dtype 4bit --attn sdpa "
                f"--device auto --model {M} --conditions ne,ev --dataset {d} --split big --batch 1" + (f" --limit {LIMIT}" if LIMIT else "") + f" > {log} 2>&1")
        sh(f"grep -E 'Error|Traceback|s/q|questions|OUT OF MEMORY|off the GPUs' -A3 {log} | tail -16; nvidia-smi --query-gpu=memory.used --format=csv,noheader")
        if rc == 0:
            break
    if rc != 0:
        ok = False; break
print("ALL PARTS DONE" if ok else "INCOMPLETE (see logs above); outputs so far are kept", flush=True)
import math
for f in sorted(glob.glob(OUT + "/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    nan = sum(any(math.isnan(x) for x in r.get("tok_logprobs", [])) for r in rows)
    empty = sum((r.get("text", "x") or "").strip() == "" for r in rows)
    gs = sum(r.get("gpu_s", 0) for r in rows) / max(len(rows), 1)
    print(f"{os.path.basename(f)}: n={len(rows)} nan={nan} empty={empty} gpu_s/q={gs:.2f} prompt_tok={sum(r.get('n_prompt', 0) for r in rows) / max(len(rows), 1):.0f}", flush=True)
print("wall time: %.0f s" % (time.time() - T0), flush=True)
sh(f"ls -la {OUT} | head -40; rm -rf {PKG} /kaggle/working/data_*")

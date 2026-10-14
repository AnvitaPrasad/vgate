"""J12 (PREREG v4c): Qwen3-VL-8B-Instruct fp16 split over two T4s. OK-VQA ne + ev on all 5,046 questions first (the registered
replication needs it), then the gold-alone (oracle) pass on all E-VQA and InfoSeek questions while time remains."""
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
EXPECTED = "e33e36023c8d"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, "roots:", ROOTS, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
if NGPU < 2:
    print("BLOCKED: Qwen3-VL-8B fp16 needs two T4s; this session has", NGPU, flush=True)
    sys.exit(0)

OUT = "/kaggle/working/J12"; os.makedirs(OUT, exist_ok=True)
M = "Qwen/Qwen3-VL-8B-Instruct"
ENV = f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/tmp/hf"
T0, BUDGET = time.time(), 8.5 * 3600  # stop well before the platform limit so that outputs are saved
ok = True
for i, (d, conds) in enumerate([("okvqa", "ne,ev"), ("evqa", "oracle"), ("infoseek", "oracle")]):
    base = (f"python -m vgate.run --data {ROOTS[d]} --images {ROOTS[d]}/images --out {OUT} --dtype float16 --attn sdpa "
            f"--device auto --model {M} --conditions {conds} --dataset {d} --split all")
    for attempt, batch in enumerate([8, 4, 2]):
        left = int(BUDGET - (time.time() - T0))
        if left < 300:
            print(f"TIME BUDGET EXHAUSTED before {d}/{conds}", flush=True); ok = False; break
        log = f"{OUT}/log_{i}_{d}_try{attempt}.txt"
        rc = sh(f"{ENV} timeout {left} {base} --batch {batch} > {log} 2>&1")
        sh(f"grep -E 'Error|Traceback|s/q|questions' -A3 {log} | tail -12; nvidia-smi --query-gpu=memory.used --format=csv,noheader")
        if rc == 0:
            break
    else:
        ok = False
    if not ok:
        break
print("ALL PARTS DONE" if ok else "INCOMPLETE (see logs above); outputs so far are kept", flush=True)
import math
for f in sorted(glob.glob(OUT + "/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    nan = sum(any(math.isnan(x) for x in r.get("tok_logprobs", [])) for r in rows)
    empty = sum((r.get("text", "x") or "").strip() == "" for r in rows)
    gs = sum(r.get("gpu_s", 0) for r in rows) / max(len(rows), 1)
    print(f"{os.path.basename(f)}: n={len(rows)} nan={nan} empty={empty} gpu_s/q={gs:.2f} prompt_tok={sum(r.get('n_prompt', 0) for r in rows) / max(len(rows), 1):.0f}", flush=True)
sh(f"ls -la {OUT} | head -40; rm -rf {PKG}")

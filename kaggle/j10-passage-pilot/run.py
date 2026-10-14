"""J10 (PREREG v3, PILOT on the dev splits): passage-level reading for Qwen3-VL-4B-Instruct.
Per question: ne; ev1_1..ev1_5 (the model reads ONE retrieved passage); ev_keep (top-5 with a keep-your-answer prompt);
pv_1..pv_5 (yes/no verification of one passage). Core conditions run first, so a time-out still leaves usable data.
GPU0: E-VQA dev, then OK-VQA dev shard 0/2.  GPU1: InfoSeek dev, then OK-VQA dev shard 1/2 (merged after download)."""
import glob, json, os, subprocess, sys, time

def sh(c):
    t = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(f"$ {c[:200]} [{time.time()-t:.0f}s rc={r.returncode}]\n{r.stdout[-3000:]}{r.stderr[-2500:]}", flush=True)

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
EXPECTED = "0ebdb068cac2"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, "roots:", ROOTS, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')

OUT = "/kaggle/working/J10"
M = "Qwen/Qwen3-VL-4B-Instruct"
CORE = "ne," + ",".join(f"ev1_{k}" for k in range(1, 6)) + ",ev_keep"
PV = ",".join(f"pv_{k}" for k in range(1, 6))
def cmd(d, conds, sub="", shard=""):
    out = os.path.join(OUT, sub) if sub else OUT
    os.makedirs(out, exist_ok=True)
    return (f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m vgate.run "
            f"--data {ROOTS[d]} --images {ROOTS[d]}/images --dataset {d} --split dev --out {out} --dtype float16 --attn sdpa "
            f"--device cuda:0 --model {M} --conditions {conds}" + (f" --shard {shard}" if shard else ""))
JOBS = {0: [cmd("evqa", CORE), cmd("okvqa", CORE, "ok0", "0/2"), cmd("evqa", PV), cmd("okvqa", PV, "ok0", "0/2")],
        1: [cmd("infoseek", CORE), cmd("okvqa", CORE, "ok1", "1/2"), cmd("infoseek", PV), cmd("okvqa", PV, "ok1", "1/2")]}
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
if NGPU < 2:  # single GPU: chain every job on GPU0, core conditions of all datasets first
    JOBS = {0: [JOBS[0][0], JOBS[1][0], JOBS[0][1], JOBS[1][1], JOBS[0][2], JOBS[1][2], JOBS[0][3], JOBS[1][3]]}
print("GPUs:", NGPU, "-> job chains:", {g: len(v) for g, v in JOBS.items()}, flush=True)
procs = []
for gpu, cmds in JOBS.items():
    chain = " ; ".join(f"HF_HOME=/tmp/hf{gpu} {c} > {OUT}/log_gpu{gpu}_{i}.txt 2>&1" for i, c in enumerate(cmds))
    procs.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu} bash -c '{chain}'", shell=True))
t = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(300)
    print(f"[{(time.time()-t)/60:.0f} min] files: {len(glob.glob(OUT + '/**/*.jsonl', recursive=True))}", flush=True)
import math
for f in sorted(glob.glob(OUT + "/**/*.jsonl", recursive=True)):
    rows = [json.loads(l) for l in open(f)]
    nan = sum(any(math.isnan(x) for x in r.get("tok_logprobs", [])) for r in rows)
    gs = sum(r.get("gpu_s", 0) for r in rows) / max(len(rows), 1)
    print(f"{f.replace(OUT + '/', '')}: n={len(rows)} nan={nan} gpu_s/q={gs:.2f} prompt_tok={sum(r.get('n_prompt', 0) for r in rows) / max(len(rows), 1):.0f}", flush=True)
sh(f"for f in {OUT}/log_*.txt; do echo == $f; grep -E 'Error|Traceback' -A6 $f | tail -12; tail -1 $f; done; rm -rf {PKG}; "
   f"rm -f {OUT}/hidden_* {OUT}/ok0/hidden_* {OUT}/ok1/hidden_*")  # closed-book hidden states already exist (J1 / J6)

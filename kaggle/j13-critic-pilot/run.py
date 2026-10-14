"""J13 (PREREG v5, PILOT on the dev splits): the released passage critic aimagelab/ReAG-Critic scores each of the top-5
retrieved passages (conditions cv_1..cv_5). A 16-question smoke run goes first so that a loading problem fails fast.
GPU0: E-VQA dev, then OK-VQA dev shard 0/2.  GPU1: InfoSeek dev, then OK-VQA dev shard 1/2 (merged after download)."""
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
EXPECTED = "5bb218ce33f3"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, "roots:", ROOTS, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')

OUT = "/kaggle/working/J13"
M = "aimagelab/ReAG-Critic"
CV = ",".join(f"cv_{k}" for k in range(1, 6))
def cmd(d, sub="", shard="", extra=""):
    out = os.path.join(OUT, sub) if sub else OUT
    os.makedirs(out, exist_ok=True)
    return (f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m vgate.run "
            f"--data {ROOTS[d]} --images {ROOTS[d]}/images --dataset {d} --split dev --out {out} --dtype float16 --attn sdpa "
            f"--device cuda:0 --model {M} --conditions {CV}" + (f" --shard {shard}" if shard else "") + extra)
rc = sh(f"CUDA_VISIBLE_DEVICES=0 HF_HOME=/tmp/hf0 {cmd('evqa', 'smoke', extra=' --limit 16 --conditions cv_1,cv_5')} 2>&1 | tail -15")
smoke = sorted(glob.glob(f"{OUT}/smoke/*.jsonl"))
for f in smoke:
    rows = [json.loads(l) for l in open(f)]
    print(os.path.basename(f), "n=%d mean p_yes=%.3f mean yes_raw=%.3f" % (len(rows), sum(r["p_yes"] for r in rows) / len(rows), sum(r["yes_raw"] for r in rows) / len(rows)), flush=True)
if rc != 0 or len(smoke) < 2:
    print("SMOKE FAILED: the critic did not load or score; stopping before the full run", flush=True)
    sys.exit(0)
JOBS = {0: [cmd("evqa"), cmd("okvqa", "ok0", "0/2")], 1: [cmd("infoseek"), cmd("okvqa", "ok1", "1/2")]}
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
if NGPU < 2:
    JOBS = {0: [JOBS[0][0], JOBS[1][0], JOBS[0][1], JOBS[1][1]]}
print("GPUs:", NGPU, "-> job chains:", {g: len(v) for g, v in JOBS.items()}, flush=True)
procs = []
for gpu, cmds in JOBS.items():
    chain = " ; ".join(f"HF_HOME=/tmp/hf{gpu} {c} > {OUT}/log_gpu{gpu}_{i}.txt 2>&1" for i, c in enumerate(cmds))
    procs.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu} bash -c '{chain}'", shell=True))
t = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(300)
    print(f"[{(time.time()-t)/60:.0f} min] files: {len(glob.glob(OUT + '/**/*.jsonl', recursive=True))}", flush=True)
for f in sorted(glob.glob(OUT + "/**/*.jsonl", recursive=True)):
    rows = [json.loads(l) for l in open(f)]
    gs = sum(r.get("gpu_s", 0) for r in rows) / max(len(rows), 1)
    print(f"{f.replace(OUT + '/', '')}: n={len(rows)} mean p_yes={sum(r['p_yes'] for r in rows) / max(len(rows), 1):.3f} gpu_s/q={gs:.2f} "
          f"prompt_tok={sum(r.get('n_prompt', 0) for r in rows) / max(len(rows), 1):.0f}", flush=True)
sh(f"for f in {OUT}/log_*.txt; do echo == $f; grep -E 'Error|Traceback' -A6 $f | tail -12; tail -1 $f; done; rm -rf {PKG}; "
   f"rm -f {OUT}/hidden_* {OUT}/*/hidden_*")

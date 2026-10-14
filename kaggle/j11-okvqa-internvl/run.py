"""J11 (PREREG v4c): InternVL3.5-4B-HF fp16. OK-VQA ne + ev on all 5,046 questions (two shards, one per GPU, merged after
download), then the gold-alone (oracle) pass on all E-VQA (GPU0) and InfoSeek (GPU1) questions."""
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
EXPECTED = "e33e36023c8d"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, "roots:", ROOTS, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')

OUT = "/kaggle/working/J11"
M = "OpenGVLab/InternVL3_5-4B-HF"
def cmd(d, conds, sub="", shard=""):
    out = os.path.join(OUT, sub) if sub else OUT
    os.makedirs(out, exist_ok=True)
    return (f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m vgate.run "
            f"--data {ROOTS[d]} --images {ROOTS[d]}/images --dataset {d} --split all --out {out} --dtype float16 --attn sdpa "
            f"--device cuda:0 --model {M} --conditions {conds}" + (f" --shard {shard}" if shard else ""))
JOBS = {0: [cmd("okvqa", "ne,ev", "ok0", "0/2"), cmd("evqa", "oracle")],
        1: [cmd("okvqa", "ne,ev", "ok1", "1/2"), cmd("infoseek", "oracle")]}
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
if NGPU < 2:  # single GPU: OK-VQA first (the registered replication needs it), then the gold-alone passes
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
import math
for f in sorted(glob.glob(OUT + "/**/*.jsonl", recursive=True)):
    rows = [json.loads(l) for l in open(f)]
    nan = sum(any(math.isnan(x) for x in r.get("tok_logprobs", [])) for r in rows)
    empty = sum((r.get("text", "x") or "").strip() == "" for r in rows)
    gs = sum(r.get("gpu_s", 0) for r in rows) / max(len(rows), 1)
    print(f"{f.replace(OUT + '/', '')}: n={len(rows)} nan={nan} empty={empty} gpu_s/q={gs:.2f} prompt_tok={sum(r.get('n_prompt', 0) for r in rows) / max(len(rows), 1):.0f}", flush=True)
sh(f"for f in {OUT}/log_*.txt; do echo == $f; grep -E 'Error|Traceback' -A6 $f | tail -12; tail -1 $f; done; rm -rf {PKG}")

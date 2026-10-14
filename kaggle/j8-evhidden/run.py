"""J8 (PREREG v2d): evidence-pass hidden states for Qwen3-VL-4B-Instruct on E-VQA, InfoSeek and OK-VQA.
Re-runs the greedy `ev` condition with --hidden_conds ev (same splits, order and batch size as J1/J4/J6) and keeps the
3L/4 layer only. Labels still come from the original runs; the regenerated answers are used to check agreement."""
import glob, json, os, subprocess, time

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
         for d in ("evqa", "infoseek", "okvqa")}  # two data mounts: vgate-data and vgate-okvqa
import sys; sys.path.insert(0, os.path.dirname(CODE))
from vgate._build import BUILD
EXPECTED = "2f8f009ff4ae"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, "roots:", ROOTS, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
for r in sorted(set(ROOTS.values())):
    sh("sha256sum " + " ".join(sorted(glob.glob(f"{r}/splits/*.txt"))))
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; python -c 'import torch;print(torch.__version__, torch.version.cuda)'")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')

OUT = "/kaggle/working/J8"; os.makedirs(OUT, exist_ok=True)
M = "Qwen/Qwen3-VL-4B-Instruct"
def cmd(d, split):
    return (f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m vgate.run "
            f"--data {ROOTS[d]} --images {ROOTS[d]}/images --dataset {d} --split {split} --out {OUT} --dtype float16 "
            f"--attn sdpa --device cuda:0 --model {M} --conditions ev --hidden_conds ev")
JOBS = {0: [cmd("evqa", "dev"), cmd("evqa", "confirm"), cmd("okvqa", "all")],
        1: [cmd("infoseek", "dev"), cmd("infoseek", "confirm")]}
procs = []
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
if NGPU < 2:  # single GPU: chain every job on GPU0
    JOBS = {0: [c for g in sorted(JOBS) for c in JOBS[g]]}
print("GPUs:", NGPU, "-> job chains:", {g: len(v) for g, v in JOBS.items()}, flush=True)
for gpu, cmds in JOBS.items():
    chain = " ; ".join(f"HF_HOME=/tmp/hf{gpu} {c} > {OUT}/log_gpu{gpu}_{i}.txt 2>&1" for i, c in enumerate(cmds))
    procs.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu} bash -c '{chain}'", shell=True))
t = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(120)
    print(f"[{(time.time()-t)/60:.0f} min] files: {sorted(os.path.basename(f) for f in glob.glob(OUT + '/*.jsonl'))}", flush=True)

meta = json.load(open(glob.glob(OUT + "/meta__*.json")[0]))
keep = sorted(meta["hidden_layers"])[1]  # 3L/4, the layer every probe uses
for f in glob.glob(OUT + "/hidden-ev_*_L*_part*.npy"):
    if f"__L{keep}_part" not in f:
        os.remove(f)
import math
for f in sorted(glob.glob(OUT + "/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    nan = sum(any(math.isnan(x) for x in r.get("tok_logprobs", [])) for r in rows)
    gs = sum(r.get("gpu_s", 0) for r in rows) / max(len(rows), 1)
    print(f"{os.path.basename(f)}: n={len(rows)} nan={nan} gpu_s/q={gs:.2f}", flush=True)
sh(f"ls -la {OUT}; for f in {OUT}/log_*.txt; do echo == $f; grep -E 'Error|Traceback|s/q|questions' -A3 $f | tail -6; done; rm -rf {PKG}")

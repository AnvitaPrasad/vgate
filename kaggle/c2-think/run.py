"""J3 C2 pilot (Thinking vs Instruct): 500 E-VQA (GPU0) + 500 InfoSeek (GPU1) dev questions (first 500 of the
frozen dev lists). Thinking checkpoint `think` (4,096-token cap, batch 4), a 2nd think sample on 200 q (label
reliability), and the Instruct noise floor `ne_tset` (Instruct NE sampled at the Thinking decoding settings)."""
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
DATA = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/splits/SPLITS.sha256", recursive=True)[0]))  # parent of splits/ and processed/
IMGS = os.path.join(DATA, "images")
import sys; sys.path.insert(0, os.path.dirname(CODE))
from vgate._build import BUILD
EXPECTED = "f445b85464dc"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh("sha256sum " + " ".join(sorted(glob.glob(f"{DATA}/splits/*.txt"))) + f"; cat {DATA}/splits/SPLITS.sha256")
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; python -c 'import torch;print(torch.__version__, torch.version.cuda)'")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')

OUT = "/kaggle/working/J3"; os.makedirs(OUT, exist_ok=True)
BASE = f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m vgate.run --data {DATA} --images {IMGS} --split dev --out {OUT} --dtype float16 --attn sdpa --device cuda:0"
JOBS = {
    g: [f"{BASE} --dataset {d} --limit 500 --model Qwen/Qwen3-VL-4B-Instruct --conditions ne_tset",
        f"{BASE} --dataset {d} --limit 500 --batch 8 --model Qwen/Qwen3-VL-4B-Thinking --conditions think",
        f"{BASE} --dataset {d} --limit 200 --batch 8 --tag _s2 --model Qwen/Qwen3-VL-4B-Thinking --conditions think"]
    for g, d in ((0, "evqa"), (1, "infoseek"))
}
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

# Summary: NaN / empty answers, s/q, token lengths per file.
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
sh(f"for f in {OUT}/log_*.txt; do echo == $f; grep -E 'Error|Traceback|s/q' -A3 $f | tail -8; done")

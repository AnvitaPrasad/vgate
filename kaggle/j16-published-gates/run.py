"""J16 (PREREG v8): two published retrieve / skip prompts run with the main reader (Qwen3-VL-4B-Instruct) on every
question of the three benchmarks (development and held-out): gate_chen (Chen et al., arXiv 2502.18023) and gate_urag
(UniversalRAG's training-free router prompt, arXiv 2504.20734). One short greedy reply per question and prompt.
GPU0: E-VQA, then OK-VQA shard 0/2.  GPU1: InfoSeek, then OK-VQA shard 1/2 (merged at the end)."""
import glob, json, os, shutil, subprocess, sys, time
from collections import Counter

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
EXPECTED = "1c72512b643e"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, "roots:", ROOTS, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')

OUT = "/kaggle/working/J16"
M = "Qwen/Qwen3-VL-4B-Instruct"
CONDS = "gate_chen,gate_urag"
def cmd(d, sub="", shard="", extra=""):
    out = os.path.join(OUT, sub) if sub else OUT
    os.makedirs(out, exist_ok=True)
    return (f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m vgate.run "
            f"--data {ROOTS[d]} --images {ROOTS[d]}/images --dataset {d} --split all --out {out} --dtype float16 --attn sdpa "
            f"--device cuda:0 --model {M} --conditions {CONDS} --hidden_conds none" + (f" --shard {shard}" if shard else "") + extra)

def summary(files):
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        n = max(len(rows), 1)
        print(f"{f.replace(OUT + '/', '')}: n={len(rows)} reads={sum(r['p_retrieve'] > 0.5 for r in rows) / n:.3f} "
              f"word_mass={sum(r['retrieve_raw'] + r['skip_raw'] for r in rows) / n:.3f} step0={sum(r['step'] == 0 for r in rows) / n:.3f} "
              f"gpu_s/q={sum(r['gpu_s'] for r in rows) / n:.2f} prompt_tok={sum(r['n_prompt'] for r in rows) / n:.0f} "
              f"replies={Counter(r['text'][:24] for r in rows).most_common(6)}", flush=True)

rc = sh(f"CUDA_VISIBLE_DEVICES=0 HF_HOME=/tmp/hf0 {cmd('evqa', 'smoke', extra=' --limit 16')} 2>&1 | tail -15")
smoke = sorted(glob.glob(f"{OUT}/smoke/*.jsonl"))
summary(smoke)
if rc != 0 or len(smoke) < 2:
    print("SMOKE FAILED: stopping before the full run", flush=True)
    sys.exit(0)
shutil.rmtree(f"{OUT}/smoke")
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
    print(f"[{(time.time()-t)/60:.0f} min] " + " ".join(f"{f.replace(OUT + '/', '')}={sum(1 for _ in open(f))}" for f in sorted(glob.glob(OUT + '/**/*.jsonl', recursive=True))), flush=True)
for sub in ("ok0", "ok1"):  # merge the two OK-VQA shards
    for f in sorted(glob.glob(f"{OUT}/{sub}/*.jsonl")):
        with open(f"{OUT}/{os.path.basename(f)}", "a") as w:
            w.write(open(f).read())
    shutil.rmtree(f"{OUT}/{sub}", ignore_errors=True)
summary(sorted(glob.glob(OUT + "/*.jsonl")))
sh(f"for f in {OUT}/log_*.txt; do echo == $f; grep -E 'Error|Traceback' -A6 $f | tail -12; tail -1 $f; done; rm -rf {PKG}")

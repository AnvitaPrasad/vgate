"""J20 (PREREG v6d, the E-VQA held-out split): two readers answer with the five passages that the image
re-ranking of PreFLMR's top-20 puts first (scripts/kbimage_rerank_export.py; PREREG v6b). One greedy evidence pass per
question; the closed-book answers and the answers with the retriever's own top-5 exist already (J4, J2).
GPU0: Qwen3-VL-4B-Instruct.  GPU1: Qwen2.5-VL-3B-Instruct."""
import glob, json, os, shutil, subprocess, sys, time

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
RR = os.path.dirname(glob.glob("/kaggle/input/**/rerank_stats.json", recursive=True)[0])  # re-ranked evidence: processed/evqa.jsonl, splits/evqa_confirm.txt
DATA = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/processed/infoseek.jsonl", recursive=True)[0]))  # the images
sys.path.insert(0, os.path.dirname(CODE))
from vgate._build import BUILD
EXPECTED = "e295da180601"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, "re-ranked evidence:", RR, "data:", DATA, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
first = json.loads(open(f"{RR}/processed/evqa.jsonl").readline())
print("stats:", open(f"{RR}/rerank_stats.json").read(), "| first record's evidence (retriever ranks):", [e.get("retriever_rank") for e in first["evidence"]], flush=True)
assert all("retriever_rank" in e for e in first["evidence"]), "this is not the re-ranked evidence file"
assert json.load(open(f"{RR}/rerank_stats.json"))["split"] == "confirm", "this is not the held-out evidence"
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')

OUT = "/kaggle/working/J20"
def cmd(model, sub="", extra=""):
    out = os.path.join(OUT, sub) if sub else OUT
    os.makedirs(out, exist_ok=True)
    return (f"PYTHONPATH={os.path.dirname(CODE)} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m vgate.run "
            f"--data {RR} --images {DATA}/images --dataset evqa --split confirm --out {out} --dtype float16 --attn sdpa "
            f"--device cuda:0 --model {model} --conditions ev --hidden_conds none" + extra)

def summary(files):
    for f in files:
        rows = [json.loads(l) for l in open(f)]
        n = max(len(rows), 1)
        print(f"{f.replace(OUT + '/', '')}: n={len(rows)} gpu_s/q={sum(r.get('gpu_s', 0) for r in rows) / n:.2f} "
              f"prompt_tok={sum(r.get('n_prompt', 0) for r in rows) / n:.0f} first replies={[r['text'][:30] for r in rows[:4]]}", flush=True)

rc = sh(f"CUDA_VISIBLE_DEVICES=0 HF_HOME=/tmp/hf0 {cmd('Qwen/Qwen3-VL-4B-Instruct', 'smoke', ' --limit 16')} 2>&1 | tail -15")
smoke = sorted(glob.glob(f"{OUT}/smoke/*.jsonl"))
summary(smoke)
if rc != 0 or not smoke:
    print("SMOKE FAILED: stopping before the full run", flush=True)
    sys.exit(0)
shutil.rmtree(f"{OUT}/smoke")
JOBS = {0: [cmd("Qwen/Qwen3-VL-4B-Instruct")], 1: [cmd("Qwen/Qwen2.5-VL-3B-Instruct")]}
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
if NGPU < 2:
    JOBS = {0: JOBS[0] + JOBS[1]}
print("GPUs:", NGPU, "-> job chains:", {g: len(v) for g, v in JOBS.items()}, flush=True)
procs = []
for gpu, cmds in JOBS.items():
    chain = " ; ".join(f"HF_HOME=/tmp/hf{gpu} {c} > {OUT}/log_gpu{gpu}_{i}.txt 2>&1" for i, c in enumerate(cmds))
    procs.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={gpu} bash -c '{chain}'", shell=True))
t = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(240)
    print(f"[{(time.time()-t)/60:.0f} min] " + " ".join(f"{f.replace(OUT + '/', '')}={sum(1 for _ in open(f))}" for f in sorted(glob.glob(OUT + '/*.jsonl'))), flush=True)
summary(sorted(glob.glob(OUT + "/*.jsonl")))
sh(f"for f in {OUT}/log_*.txt; do echo == $f; grep -E 'Error|Traceback' -A6 $f | tail -12; tail -1 $f; done; rm -rf {PKG}")

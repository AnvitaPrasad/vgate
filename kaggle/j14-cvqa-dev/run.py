"""J14 CVQA, a fourth benchmark (PREREG v7), development part. Qwen3-VL-4B-Instruct, one shard per T4.
1. questions and images from HF `afaji/cvqa` (images go to /tmp and are not kept);
2. entity-name pass (`ent`) on all 10,374 questions -> retrieval queries;
3. dense retrieval (bge-m3) over the M4-RAG Wikipedia corpus -> top-5 evidence for every question;
4. closed-book (`ne`) and evidence (`ev`) passes on the development split only (2,000 questions).
The held-out split is read by a later job (J15) from the records written here."""
import glob, json, os, shutil, subprocess, sys, time

def sh(c, check=False):
    t = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(f"$ {c[:220]} [{time.time()-t:.0f}s rc={r.returncode}]\n{r.stdout[-3500:]}{r.stderr[-2500:]}", flush=True)
    assert not check or r.returncode == 0, "step failed"

# Kaggle flattens the single-folder code zip into /kaggle/input/datasets/<user>/vgate-code/: rebuild the package.
SRC = os.path.dirname((glob.glob("/kaggle/input/**/vgate/run.py", recursive=True)
                       or glob.glob("/kaggle/input/**/vgate-code/run.py", recursive=True))[0])
PKG = "/kaggle/working/pkg"; os.makedirs(PKG, exist_ok=True)
subprocess.run(f"rm -rf {PKG}/vgate && cp -r {SRC} {PKG}/vgate", shell=True, check=True)
sys.path.insert(0, PKG)
from vgate._build import BUILD
EXPECTED = "e54d088f36fe"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; python -c 'import torch;print(torch.__version__, torch.version.cuda)'; df -h /tmp /kaggle/working | tail -2")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')

OUT, RAW, IMGS = "/kaggle/working/J14", "/tmp/cvqa_raw", "/tmp/cvqa_img"
DATA = f"{OUT}/data"; os.makedirs(OUT, exist_ok=True)
M = "Qwen/Qwen3-VL-4B-Instruct"; MN = M.split("/")[-1]
PY = f"PYTHONPATH={PKG} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m"
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)

from huggingface_hub import snapshot_download
t = time.time()
snapshot_download("afaji/cvqa", repo_type="dataset", local_dir=f"{RAW}/cvqa", allow_patterns=["data/*.parquet"])
snapshot_download("davidanugraha/M4-RAG", repo_type="dataset", local_dir=f"{RAW}/m4rag", allow_patterns=["cvqa/*.parquet"])
print(f"downloads {time.time()-t:.0f}s", flush=True)
sh(f"{PY} vgate.cvqa questions --raw {RAW}/cvqa --images {IMGS} --data {DATA}", check=True)
shutil.rmtree(f"{RAW}/cvqa")

def passes(split, conds, extra=""):
    """One shard per GPU into OUT/s<g>; blocks until both finish."""
    procs = []
    for g in range(NGPU):
        c = (f"{PY} vgate.run --data {DATA} --images {IMGS} --dataset cvqa --split {split} --out {OUT}/s{g} --dtype float16 --attn sdpa "
             f"--device cuda:0 --model {M} --conditions {conds} --shard {g}/{NGPU} {extra}")
        procs.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} HF_HOME=/tmp/hf{g} bash -c '{c} >> {OUT}/log_gpu{g}_{conds.replace(',', '-')}.txt 2>&1'", shell=True))
    t = time.time()
    while any(p.poll() is None for p in procs):
        time.sleep(120)
        print(f"[{conds} {(time.time()-t)/60:.0f} min] " + " ".join(f"{os.path.basename(f)}={sum(1 for _ in open(f))}" for f in sorted(glob.glob(f"{OUT}/s*/*.jsonl"))), flush=True)
    sh(f"for f in {OUT}/log_gpu*_{conds.replace(',', '-')}.txt; do echo == $f; grep -E 'Error|Traceback|s/q|missing' -A3 $f | tail -8; done")
    assert all(p.returncode == 0 for p in procs), f"pass {conds} failed"

passes("all", "ent", "--hidden_conds none")
dirs = ",".join(f"{OUT}/s{g}" for g in range(NGPU))
sh(f"{PY} vgate.cvqa retrieve --data {DATA} --ent {dirs} --corpus {RAW}/m4rag --devices {','.join(f'cuda:{g}' for g in range(NGPU))}", check=True)
recs = [json.loads(l) for l in open(f"{DATA}/processed/cvqa.jsonl")]
assert all(len(r["evidence"]) == 5 and len(r["retr_scores"]) == 20 for r in recs), "retrieval incomplete"
import collections
print("evidence languages (top-1):", collections.Counter(r["evidence"][0]["lang"] for r in recs).most_common(12), flush=True)
print("mean evidence words:", sum(len(e["text"].split()) for r in recs for e in r["evidence"]) / (5 * len(recs)), flush=True)
for r in recs[:3]:
    print(json.dumps(dict(q=r["query"], opts=r["options"], ans=r["answers"], ev=[(e["lang"], e["title"], e["score"], e["text"][:80]) for e in r["evidence"]]), ensure_ascii=False)[:1200], flush=True)

passes("dev", "ne,ev", "--batch 4")  # evidence in some scripts is token-heavy: two prompts per evidence batch

# Merge the shards: answers by concatenation, hidden-state parts renumbered.
for g in range(NGPU):
    for f in sorted(glob.glob(f"{OUT}/s{g}/*.jsonl")):
        with open(f"{OUT}/{os.path.basename(f)}", "a") as w:
            w.write(open(f).read())
    for f in sorted(glob.glob(f"{OUT}/s{g}/hidden*")):
        b = os.path.basename(f)
        shutil.move(f, f"{OUT}/" + (b.replace("qids0", f"qids{g}") if "qids" in b else b.replace("_part0", f"_part{g}")))
    for f in glob.glob(f"{OUT}/s{g}/meta__*.json"):
        shutil.copy(f, OUT)
    shutil.rmtree(f"{OUT}/s{g}")
shutil.rmtree(PKG, ignore_errors=True)

# Summary: counts, NaN / empty answers, s/q, answer letters, accuracy of the two passes on the development split.
import math
from collections import Counter
gold = {r["qid"]: r for r in recs}
for f in sorted(glob.glob(OUT + "/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    nan = sum(any(math.isnan(x) for x in r.get("tok_logprobs", [])) for r in rows)
    empty = sum((r.get("text", "x") or "").strip() == "" for r in rows)
    gs = sum(r.get("gpu_s", 0) for r in rows) / max(len(rows), 1)
    npr = sum(r.get("n_prompt", 0) for r in rows) / max(len(rows), 1)
    line = f"{os.path.basename(f)}: n={len(rows)} unique={len({r['qid'] for r in rows})} nan={nan} empty={empty} gpu_s/q={gs:.2f} prompt_tok={npr:.0f}"
    if not f.endswith("ent.jsonl"):
        first = Counter((r["text"].strip()[:1] or "_") for r in rows)
        acc = sum(r["text"].strip()[:1] == gold[r["qid"]]["answers"][0] for r in rows) / max(len(rows), 1)
        line += f" first_char={dict(first.most_common(6))} first-char accuracy={acc:.4f}"
    print(line, flush=True)
print("files:", sorted(os.listdir(OUT)), flush=True)

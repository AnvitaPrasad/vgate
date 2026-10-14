"""J18b: reruns the 72 questions on which J18 returned NaN (half precision), one per batch in float32. Derived from J18 CVQA (PREREG v7d), a second reader: closed-book and evidence passes of Qwen2.5-VL-3B-Instruct on every CVQA question
(development and held-out), with the records, evidence and splits written by J14b (the evidence was retrieved with the
main reader's entity names and is the same for both readers). No accuracy is printed here."""
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
EXPECTED = "e295da180601"
print("mounted vgate build:", BUILD, "expected:", EXPECTED, flush=True)
assert BUILD == EXPECTED, f"stale code mount: got {BUILD}, expected {EXPECTED}"
sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; python -c 'import torch;print(torch.__version__, torch.version.cuda)'; df -h /tmp /kaggle/working | tail -2")
sh('pip install -q "transformers==5.16.1" accelerate 2>&1 | tail -2; python -c "import transformers;print(transformers.__version__)"')

OUT, RAW, IMGS = "/kaggle/working/J18b", "/tmp/cvqa_raw", "/tmp/cvqa_img"
DATA = "/kaggle/working/J18b_data"; os.makedirs(OUT, exist_ok=True)
M = "Qwen/Qwen2.5-VL-3B-Instruct"; MN = M.split("/")[-1]
PY = f"PYTHONPATH={PKG} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m"
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)

# Records with evidence and the frozen splits come from J14's output; the images are read again from the source.
src = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/processed/cvqa.jsonl", recursive=True)[0]))
shutil.copytree(src, DATA)
sh(f"ls -la {DATA}/processed {DATA}/splits; wc -l {DATA}/splits/*.txt; sha256sum {DATA}/splits/*.txt {DATA}/processed/cvqa.jsonl")
from huggingface_hub import snapshot_download
t = time.time()
snapshot_download("afaji/cvqa", repo_type="dataset", local_dir=f"{RAW}/cvqa", allow_patterns=["data/*.parquet"])
print(f"download {time.time()-t:.0f}s", flush=True)
sh(f"PYTHONPATH={PKG} python -c \"from vgate import cvqa; cvqa.questions('{RAW}/cvqa', '{IMGS}')\"", check=True)
shutil.rmtree(f"{RAW}/cvqa")
recs = [json.loads(l) for l in open(f"{DATA}/processed/cvqa.jsonl")]
assert all(len(r["evidence"]) == 5 for r in recs) and all(os.path.exists(f"{IMGS}/{r['image_file']}") for r in recs), "records or images incomplete"

FIX = ["5865687714278936238_0", "5865687714278936238_1", "5865743574277362202_0", "5865743574277362202_1", "5865743574277362202_2", "5865743584275306627_0", "5865743584275306627_1", "5865743594274065189_0", "5865743594274065189_1", "5865745404274872669_0", "5865745404274872669_1", "5865745404274872669_2", "5865745404275445480_0", "5865745404275445480_1", "5865745404275445480_2", "5865752674279498040_0", "5865752674279498040_1", "5865752674279498040_2", "5865752684273451179_0", "5865752684273451179_1", "5865776454272237826_0", "5865776454272237826_1", "5865910674276486487_0", "5865910674276486487_1", "5865910674276486487_2", "5865926694273721134_0", "5865926694278771250_0", "5865926694278771250_1", "5865939214273399065_0", "5865939214273399065_1", "5865939214273399065_2", "5865939224274921551_0", "5865939224274921551_1", "5865939224274921551_2", "5865939244275678287_0", "5865939244275678287_1", "5865939244275678287_2", "5865939244276965786_0", "5865939244276965786_1", "5865939244276965786_2", "5865939264271157600_0", "5865939264271157600_2", "5865939264272065899_0", "5865939264272065899_1", "5865939264273932460_0", "5865939264275048390_0", "5865939264275048390_1", "5865939264275048390_2", "5865940604274051658_0", "5865940604274051658_1", "5865941064275084723_0", "5865941064275084723_1", "5865941064275084723_2", "5865942094271232930_0", "5865942094271232930_1", "5865942524273348507_0", "5865943144271982260_0", "5865943144276650586_0", "5865943144276650586_1", "5878749247629871703_0", "5920593834278341721_0", "5920593834278341721_1", "5920942824273086457_0", "5920942824273086457_1", "5920943064278181597_0", "5920943064278181597_1", "5920948894275217257_0", "5920948894275217257_1", "5920948894275217257_2", "5922885495416557009_0", "6028345774271126672_0", "6028345774271126672_1"]
open(f"{DATA}/splits/cvqa_fix.txt", "w").write("\n".join(FIX) + "\n")
c = (f"{PY} vgate.run --data {DATA} --images {IMGS} --dataset cvqa --split fix --out {OUT} --dtype float32 --attn sdpa "
     f"--device auto --model {M} --conditions ne,ev --batch 1")  # version 1 ran on one T4 and ran out of memory in the evidence pass
sh(f"HF_HOME=/tmp/hf0 bash -c '{c} > {OUT}/log_fix.txt 2>&1'; tail -5 {OUT}/log_fix.txt; grep -E 'Error|Traceback' -A5 {OUT}/log_fix.txt | tail -12")
shutil.rmtree(PKG, ignore_errors=True); shutil.rmtree(DATA, ignore_errors=True)
import math
for f in sorted(glob.glob(OUT + "/*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    nan = sum(any(math.isnan(x) for x in r.get("tok_logprobs", [])) for r in rows)
    print(f"{os.path.basename(f)}: n={len(rows)} unique={len({r['qid'] for r in rows})} nan={nan} first replies={[r['text'][:12] for r in rows[:8]]}", flush=True)

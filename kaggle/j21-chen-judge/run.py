"""J21 (PREREG v11b): an open judge for the released answers of Chen et al. (arXiv:2502.18023), Mix set.
Their judge prompt, unchanged (evaluation/src/vqa_eval.py, qwen_evaluate), is sent to Qwen2.5-14B-Instruct (4-bit, greedy)
instead of Qwen-Max: the system prompt, then the processed model output and the reference answers. The first line of the
reply is parsed as a number, as in their code; anything else scores 0. One worker per T4. No accuracy is printed here."""
import glob, json, os, subprocess, sys, time

LIMIT = 0  # judge inputs per worker; 0 = all (6,000 in total). A positive value runs a smoke test only.
# Version 1 judged 5,992 answers; worker 0 ran out of memory on its last batch (the eight longest replies, batch of 8).
# Version 2 judges only those eight, one at a time, with the same judge, prompt and decoding. Empty list = everything.
ONLY = [1542, 2742, 4826, 5156, 5582, 5674, 5752, 5778]  # line numbers (0-based) in judge_inputs.jsonl
MODEL = "Qwen/Qwen2.5-14B-Instruct"
OUT = "/kaggle/working/J21"
WORKER = r'''
import json, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
part, nparts, limit, src, out, model_id = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
BS = int(sys.argv[7]) if len(sys.argv) > 7 else 8
SYSTEM = """请你参考下面的参考答案，评估模型输出，并给出0到1之间的打分。
- 0表示模型输出完全错误，1表示模型输出正确。
- 你也可以打出0到1之间的小数。参考答案可能会有多个（每行一个），模型输出只会有一个。
- 请只输出数字，不要输出其他任何内容。
示例输出：0.8

"""
USER = """模型输出：
{pred}

参考答案
{answer}

输出：
"""
rows = [json.loads(l) for l in open(src)]
rows = [r for i, r in enumerate(rows) if i % nparts == part]
if limit:
    rows = rows[:limit]
tok = AutoTokenizer.from_pretrained(model_id); tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda:0", quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)).eval()
rows.sort(key=lambda r: len(r["pred"]))  # similar lengths in a batch
t0, done = time.time(), 0
with open(out, "w") as f:
    for i in range(0, len(rows), BS):
        b = rows[i:i + BS]
        texts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                          {"role": "user", "content": USER.format(pred=r["pred"], answer="\n".join(map(str, r["answers"])))}],
                                         tokenize=False, add_generation_prompt=True) for r in b]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to("cuda:0")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
        for r, g in zip(b, gen):
            raw = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                score = float(raw.split("\n")[0].strip())   # as their code; a reply that is not a number scores 0
            except ValueError:
                score = 0.0
            f.write(json.dumps(dict(reader=r["reader"], cond=r["cond"], row=r["row"], raw=raw, score=score), ensure_ascii=False) + "\n")
        done += len(b)
        if (i // 8) % 50 == 0:
            print(f"part {part}: {done}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
print(f"part {part} done: {done} in {time.time() - t0:.0f}s", flush=True)
'''

def sh(c):
    t = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(f"$ {c[:200]} [{time.time()-t:.0f}s rc={r.returncode}]\n{r.stdout[-2500:]}{r.stderr[-2500:]}", flush=True)
    return r.returncode

SRC = glob.glob("/kaggle/input/**/judge_inputs.jsonl", recursive=True)[0]
if ONLY:
    lines = open(SRC).read().splitlines()
    SRC = "/tmp/judge_inputs_only.jsonl"; open(SRC, "w").write("\n".join(lines[i] for i in ONLY) + "\n")
print("judge inputs:", SRC, sum(1 for _ in open(SRC)), "| model:", MODEL, "| limit:", LIMIT, flush=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh('pip install -q -U bitsandbytes accelerate 2>&1 | tail -2; python -c "import transformers, bitsandbytes; print(transformers.__version__, bitsandbytes.__version__)"')
os.makedirs(OUT, exist_ok=True)
open("/tmp/worker.py", "w").write(WORKER)
sh(f"HF_HOME=/tmp/hf python -c \"from huggingface_hub import snapshot_download; print(snapshot_download('{MODEL}'))\" 2>&1 | tail -1; df -h /tmp | tail -1")  # one download, shared by both workers
NGPU = int(subprocess.run("nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True).stdout.strip() or 1)
BS = 8
if ONLY:
    NGPU, BS = 1, 1
procs = [subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} HF_HOME=/tmp/hf python /tmp/worker.py {g} {NGPU} {LIMIT} {SRC} {OUT}/judge_part{'_rest' if ONLY else ''}{g}.jsonl {MODEL} {BS} "
                          f"> {OUT}/log_gpu{g}.txt 2>&1", shell=True) for g in range(NGPU)]
for p in procs:
    p.wait()
for g in range(NGPU):
    sh(f"grep -E 'Error|Traceback' -A6 {OUT}/log_gpu{g}.txt | tail -12; tail -2 {OUT}/log_gpu{g}.txt")
rows = [json.loads(l) for f in sorted(glob.glob(f"{OUT}/judge_part*.jsonl")) for l in open(f)]
bad = sum(1 for r in rows if not r["raw"].strip()[:1].isdigit())
print("judged:", len(rows), "| replies that do not start with a digit:", bad, "| first raw replies:", [r["raw"] for r in rows[:6]], flush=True)

"""J17 (PREREG v8): the retrieval decision of the released ReflectiVA checkpoint (aimagelab/ReflectiVA, Apache-2.0;
Cocchi et al., arXiv 2411.16863) on our questions. As in the release (rag_evaluation/infoseek/release_retrieval.py):
conversation template llama_3_1, user turn "<image>\n\n" + question, first generated token [Retrieval] (128251) or
[No Retrieval] (128250). The release samples that token at temperature 0.2; here it is read greedily and the two
probabilities are kept. fp16, one model split over both T4s. LIMIT > 0 runs a smoke test only."""
import glob, json, os, subprocess, sys, time

LIMIT = 0           # questions per benchmark; 0 = all (development and held-out). The smoke run (16) passed on 2026-10-06.
LIBS, REPO, OUT = "/tmp/rvlibs", "/tmp/ReflectiVA", "/kaggle/working/J17"
WORKER = r'''
import glob, json, os, sys, time
import torch
from PIL import Image
os.environ["TOKENIZER_PATH"] = "aimagelab/ReflectiVA"   # llava/conversation.py loads the tokenizer at import
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images

RET, NORET = 128251, 128250
limit, out = int(sys.argv[1]), sys.argv[2]
tokenizer, model, image_processor, _ = load_pretrained_model("aimagelab/ReflectiVA", None, "llava_llama_3.1",
                                                             max_memory={0: "9GiB", 1: "13GiB"} if torch.cuda.device_count() > 1 else None)
model.eval()
print("tokens:", tokenizer.convert_ids_to_tokens([NORET, RET]), "devices:", sorted({str(p.device) for p in model.parameters()}), flush=True)
assert tokenizer.convert_ids_to_tokens([NORET, RET]) == ["[No Retrieval]", "[Retrieval]"], "unexpected special tokens"
dev = model.device
for d in ("evqa", "infoseek", "okvqa"):
    root = os.path.dirname(os.path.dirname(glob.glob(f"/kaggle/input/**/processed/{d}.jsonl", recursive=True)[0]))
    recs = [json.loads(l) for l in open(f"{root}/processed/{d}.jsonl")]
    if limit:
        recs = recs[:limit]
    idx = {}
    for dp, _, fs in os.walk(f"{root}/images"):
        for f in fs:
            idx.setdefault(f, os.path.join(dp, f))
    path = f"{out}/{d}__ReflectiVA__gate.jsonl"
    done = {json.loads(l)["qid"] for l in open(path)} if os.path.exists(path) else set()
    t0, n = time.time(), 0
    with open(path, "a") as f:
        for r in recs:
            if r["qid"] in done:
                continue
            conv = conv_templates["llama_3_1"].copy()
            conv.append_message(conv.roles[0], f"{DEFAULT_IMAGE_TOKEN}\n\n" + r["question"])
            conv.append_message(conv.roles[1], None)
            ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")[1:].unsqueeze(0).to(dev)
            im = Image.open(idx[os.path.basename(r["image_file"])]).convert("RGB")
            px = process_images([im], image_processor, model.config)[0].unsqueeze(0).to(dtype=torch.float16, device=dev)
            with torch.inference_mode():
                o = model.generate(ids, images=px, image_sizes=im.size, do_sample=False, num_beams=1, max_new_tokens=1, use_cache=True,
                                   output_scores=True, return_dict_in_generate=True)
            p = torch.softmax(o.scores[0][0].float(), -1)
            a, b, top = float(p[RET]), float(p[NORET]), int(p.argmax())
            f.write(json.dumps(dict(qid=r["qid"], p_retrieve=round(a / (a + b + 1e-12), 5), retrieve_raw=round(a, 5), skip_raw=round(b, 5),
                                    top=top, text=tokenizer.convert_ids_to_tokens([top])[0], n_prompt=int(ids.shape[1]), cond="gate")) + "\n")
            n += 1
            if n % 200 == 0:
                f.flush(); print(f"[{d}] {n}/{len(recs) - len(done)} {(time.time() - t0) / n:.2f}s/q", flush=True)
    rows = [json.loads(l) for l in open(path)]
    m = max(len(rows), 1)
    print(f"{d}: n={len(rows)} retrieves={sum(x['p_retrieve'] > 0.5 for x in rows) / m:.3f} first token is one of the two={sum(x['top'] in (RET, NORET) for x in rows) / m:.3f} "
          f"mass on the two={sum(x['retrieve_raw'] + x['skip_raw'] for x in rows) / m:.3f} s/q={(time.time() - t0) / max(n, 1):.2f}", flush=True)
'''

def sh(c):
    t = time.time(); r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(f"$ {c[:200]} [{time.time()-t:.0f}s rc={r.returncode}]\n{r.stdout[-3500:]}{r.stderr[-3500:]}", flush=True)
    return r

os.makedirs(OUT, exist_ok=True)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; python -V; python -c 'import torch, numpy; print(torch.__version__, numpy.__version__)'")
sh(f"git clone --depth 1 https://github.com/aimagelab/ReflectiVA {REPO} && cd {REPO} && git rev-parse HEAD")
# The fork pins transformers 4.43.1, whose tokenizers wheel does not build on this Python (3.13). The smoke run loaded and ran
# the checkpoint with the image's own transformers: every first token was one of the two retrieval tokens (mass 0.97).
sh("python -c 'import transformers, accelerate; print(transformers.__version__, accelerate.__version__)'")
open("/tmp/worker.py", "w").write(WORKER)
os.makedirs(LIBS, exist_ok=True)  # only for packages the fork asks for and the image lacks
env = f"PYTHONPATH={LIBS}:{REPO} HF_HOME=/tmp/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
for attempt in range(8):  # the fork may import packages that are not installed: install what is named and retry
    r = sh(f"{env} python /tmp/worker.py {LIMIT} {OUT} 2>&1 | grep -v -i -E 'warn|deprecat|UNEXPECTED|Loading weights' | tail -80")
    miss = [l.split("'")[1] for l in r.stdout.splitlines() if "ModuleNotFoundError" in l and "'" in l]
    if not miss:
        break
    name = {"s2wrapper": "git+https://github.com/bfshi/scaling_on_scales.git", "open_clip": "open_clip_torch"}.get(miss[-1].split(".")[0], miss[-1].split(".")[0])
    sh(f"pip install -q --target {LIBS} '{name}' 2>&1 | tail -2")
print("files:", {os.path.basename(f): sum(1 for _ in open(f)) for f in sorted(glob.glob(OUT + '/*.jsonl'))}, flush=True)

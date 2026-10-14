"""Run one model over one dataset split for a list of conditions. Resumable: rows are keyed by qid.

  python -m vgate.run --dataset evqa --split dev --model Qwen/Qwen3-VL-4B-Instruct \
      --images /path/to/images --conditions ne,ne_sample,ent,ent_sample,ptrue,ev,oracle,para \
      --out runs/g1 [--limit 32] [--dtype float16] [--batch 8]

Outputs <out>/<dataset>__<model>__<condition>.jsonl (+ hidden_<dataset>__<model>__L<i>.npy for `ne`).
Condition semantics (prompt text in prompts.py):
  ne          greedy, no evidence  (+ token logprobs/entropy, hidden states)  -> label c_without
  ev          greedy, top-5 retrieved evidence                                 -> label c_with
  oracle      greedy, gold evidence (upper bound only; never a gate input)
  ne_sample   k samples, no evidence (consistency features)
  ent/ent_sample  entity-name greedy + k samples (recognition-consistency feature)
  ptrue       P(yes) that the ne answer is correct
  para        greedy on a perturbed (lower-cased) question (perturbation-consistency feature)
  think       Thinking checkpoint, model's own sampling config, long budget (C2 only)
  ne_tset     Instruct NE sampled at the Thinking checkpoint's decoding settings (C2 noise floor)
"""
import argparse
import json
import os
import pathlib
import time

import numpy as np

from . import prompts
from .models import VLM, load_image

SAMPLE = dict(temperature=0.7, top_p=0.95, top_k=50)  # frozen feature-sampling config
THINK_SAMPLE = dict(temperature=1.0, top_p=0.95, top_k=20)  # Qwen3-VL-*-Thinking generation_config.json (C2 noise floor)


def resolve_images(root):
    """basename -> path index (E-VQA/InfoSeek archives unpack to different folder layouts)."""
    idx = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            idx.setdefault(f, os.path.join(dp, f))
    return idx


def load_split(data_dir, dataset, split, limit):
    recs = {r["qid"]: r for r in map(json.loads, open(pathlib.Path(data_dir) / "processed" / f"{dataset}.jsonl"))}
    if split == "all":
        qids = list(recs)
    else:
        qids = (pathlib.Path(data_dir) / "splits" / f"{dataset}_{split}.txt").read_text().split()
    return [recs[q] for q in qids[:limit or None]]


def done_qids(path):
    if not path.exists():
        return set()
    return {json.loads(l)["qid"] for l in open(path) if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--conditions", default="ne,ev,oracle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--think_tokens", type=int, default=4096)
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _rerun")
    ap.add_argument("--shard", default="", help="i/n: keep every n-th question starting at i (to split a split over GPUs)")
    ap.add_argument("--hidden_conds", default="ne", help="conditions whose last-prompt-token hidden states are saved "
                    "(ne -> hidden_*; any other condition c -> hidden-c_*)")
    ap.add_argument("--shuffle_seed", type=int, default=-1, help="permute question order (batch-order check)")
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = load_split(a.data, a.dataset, a.split, a.limit)
    img_idx = resolve_images(a.images)
    missing = [r["qid"] for r in recs if os.path.basename(r["image_file"]) not in img_idx]
    print(f"{a.dataset}/{a.split}: {len(recs)} questions, {len(missing)} missing images", flush=True)
    recs = [r for r in recs if r["qid"] not in set(missing)]
    if a.shard:
        i, n = map(int, a.shard.split("/"))
        recs = recs[i::n]
    if a.shuffle_seed >= 0:
        import random
        random.Random(a.shuffle_seed).shuffle(recs)
    vlm = VLM(a.model, dtype=a.dtype, attn=a.attn, device=a.device)
    mname = a.model.split("/")[-1]
    L = vlm.n_layers
    hid_layers = sorted({L // 2, (3 * L) // 4, L - 1})
    meta = dict(model=a.model, dtype=a.dtype, attn=a.attn, prompt_id=prompts.PROMPT_ID, k=a.k,
                sample=SAMPLE, max_area=512 * 512, n_layers=L, hidden_layers=hid_layers)
    (out / f"meta__{a.dataset}__{mname}.json").write_text(json.dumps(meta, indent=1))

    ne_answers = {}
    hid_conds = set(a.hidden_conds.split(","))
    for cond in a.conditions.split(","):
        path = out / f"{a.dataset}__{mname}__{cond}{a.tag}.jsonl"
        if cond in ("ptrue", "ev_keep"):  # need the greedy NE answers
            ne_path = out / f"{a.dataset}__{mname}__ne.jsonl"
            ne_answers = {d["qid"]: d["text"] for d in map(json.loads, open(ne_path))}
        todo = [r for r in recs if r["qid"] not in done_qids(path)]
        print(f"[{cond}] {len(todo)} to run", flush=True)
        long_prompt = cond in ("ev", "oracle", "think", "ev_keep") or cond.startswith(("pv_", "cv_"))  # pv / cv: full-vocabulary logits
        bs = max(1, a.batch // 2) if long_prompt else a.batch
        hid_buf = {li: [] for li in hid_layers}
        hid_q = []
        t_start = time.time()
        with open(path, "a") as f:
            for b in range(0, len(todo), bs):
                chunk = todo[b:b + bs]
                ims = [load_image(img_idx[os.path.basename(r["image_file"])]) for r in chunk]
                if cond in prompts.GATE_SPECS:  # a published retrieve / skip prompt, read from the first word of the reply
                    g = prompts.GATE_SPECS[cond]
                    res = vlm.first_token(ims, [prompts.build(cond, r) for r in chunk], g["retrieve"], g["skip"], image_pos=g["image_pos"])
                    rows = [dict(qid=r["qid"], **x) for r, x in zip(chunk, res)]
                elif cond == "ptrue" or cond.startswith(("pv_", "cv_")):  # P(yes) at the first generated position
                    texts = [prompts.build(cond, r, answer=ne_answers.get(r["qid"])) for r in chunk]
                    res = vlm.p_yes(ims, texts, system=prompts.CV_SYSTEM if cond.startswith("cv_") else None)
                    rows = [dict(qid=r["qid"], **x) for r, x in zip(chunk, res)]
                else:
                    texts = [prompts.build(cond, r, answer=ne_answers.get(r["qid"]), k=a.k) for r in chunk]
                    sample = (SAMPLE if cond.endswith("_sample") else "model" if cond == "think"
                              else THINK_SAMPLE if cond == "ne_tset" else None)
                    n = a.k if cond.endswith("_sample") else 1
                    gen = lambda: vlm.generate(
                        ims, texts, max_new_tokens=a.think_tokens if cond == "think" else a.max_new_tokens,
                        sample=sample, n=n, hidden_layers=hid_layers if cond in hid_conds and not a.tag else (),
                        want_logits=cond != "think")
                    try:
                        res, hid = gen()
                    except RuntimeError as e:  # a prompt that does not fit beside the weights (PREREG v13, a 4-bit 32B reader on two T4s)
                        if "out of memory" not in str(e).lower():
                            raise
                        import gc
                        import torch
                        gc.collect(); torch.cuda.empty_cache()
                        try:
                            res, hid = gen()
                        except RuntimeError as e2:
                            if "out of memory" not in str(e2).lower():
                                raise
                            gc.collect(); torch.cuda.empty_cache()
                            print(f"  [{cond}] OUT OF MEMORY, no row written for {[r['qid'] for r in chunk]}", flush=True)
                            continue
                    if n == 1:
                        rows = [dict(qid=r["qid"], **x) for r, x in zip(chunk, res)]
                    else:
                        rows = [dict(qid=r["qid"], samples=[x["text"] for x in res[i * n:(i + 1) * n]],
                                     seq_logprobs=[round(sum(x["tok_logprobs"]), 4) for x in res[i * n:(i + 1) * n]],
                                     n_prompt=res[i * n]["n_prompt"], n_gen=sum(x["n_gen"] for x in res[i * n:(i + 1) * n]),
                                     gpu_s=round(sum(x["gpu_s"] for x in res[i * n:(i + 1) * n]), 5))
                                for i, r in enumerate(chunk)]
                    if hid:
                        for li in hid_layers:
                            hid_buf[li].append(hid[li].numpy().astype(np.float16))
                        hid_q += [r["qid"] for r in chunk]
                for row in rows:
                    row["cond"] = cond + a.tag
                    f.write(json.dumps(row) + "\n")
                f.flush()
                if (b // bs) % 20 == 0:
                    el = time.time() - t_start
                    print(f"  [{cond}] {b + len(chunk)}/{len(todo)}  {el / (b + len(chunk)):.2f}s/q", flush=True)
        if hid_q:  # hidden states for resumed partial runs are appended as separate parts
            hp = "hidden" if cond == "ne" else f"hidden-{cond}"
            part = len(list(out.glob(f"{hp}_{a.dataset}__{mname}__qids*.json")))
            (out / f"{hp}_{a.dataset}__{mname}__qids{part}.json").write_text(json.dumps(hid_q))
            for li in hid_layers:
                np.save(out / f"{hp}_{a.dataset}__{mname}__L{li}_part{part}.npy", np.concatenate(hid_buf[li]))


if __name__ == "__main__":
    main()

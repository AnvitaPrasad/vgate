"""Batched HF inference for Qwen2.5-VL / Qwen3-VL (and other AutoModelForImageTextToText VLMs).

Returns, per prompt: generated text, raw-logit token logprobs and entropies (from `output_logits`, i.e.
before any sampling warpers), prompt/generated token counts, and the batch's wall time for the cost
ledger. Hidden states of the LAST PROMPT TOKEN at chosen decoder layers are captured with forward hooks
during prefill only (output_hidden_states would materialise every layer x every position).
"""
import time

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MAX_AREA = 512 * 512  # frozen: caps image tokens at ~256 (Qwen3-VL) / ~334 (Qwen2.5-VL)


def prepare_image(im, max_area=MAX_AREA):
    """RGB + the frozen area cap (used for files and for images uploaded to the live demo)."""
    im = im.convert("RGB")
    w, h = im.size
    if w * h > max_area:
        s = (max_area / (w * h)) ** 0.5
        im = im.resize((max(28, int(w * s)), max(28, int(h * s))), Image.BICUBIC)
    return im


def load_image(path, max_area=MAX_AREA):
    return prepare_image(Image.open(path), max_area)


def _decoder_layers(model):
    """The text decoder's layer list (largest ModuleList named *layers outside the vision tower)."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers") and "visual" not in name \
                and "vision" not in name and (best is None or len(mod) > len(best[1])):
            best = (name, mod)
    return best[1]


class VLM:
    def __init__(self, model_id, dtype="float16", attn="sdpa", device="cuda:0"):
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        kw = dict(device_map=device)
        self.multi = device == "auto"
        if self.multi:  # one model split over all visible GPUs (Qwen3-VL-8B fp16 does not fit a single T4)
            kw = dict(device_map="auto", max_memory={i: "7GiB" if i == 0 else "13GiB"
                                                     for i in range(torch.cuda.device_count())})
            device = "cuda:0"  # inputs go to the first device; accelerate moves activations between GPUs
        if dtype == "4bit":  # a reader too large for two T4s in half precision: NF4 weights, half-precision compute (PREREG v13)
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            if self.multi:  # weights per card; what is left is for the prompt's activations (VGATE_MAX_MEMORY="10GiB,11500MiB" overrides)
                import os
                mm = os.environ.get("VGATE_MAX_MEMORY", "10GiB,13GiB").split(",")
                kw["max_memory"] = {i: mm[min(i, len(mm) - 1)] for i in range(torch.cuda.device_count())}
            dtype = "float16"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=getattr(torch, dtype), attn_implementation=attn, **kw).eval()
        placed = set(map(str, getattr(self.model, "hf_device_map", {}).values()))
        assert not placed & {"cpu", "disk"}, f"part of the model was placed off the GPUs: {sorted(placed)}"
        self.device = device
        self.layers = _decoder_layers(self.model)
        self.n_layers = len(self.layers)
        tok = self.processor.tokenizer
        self.eos_ids = {i for i in [tok.eos_token_id, tok.pad_token_id,
                                    tok.convert_tokens_to_ids("<|im_end|>")] if i is not None and i >= 0}
        self.yes_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ["yes", "Yes", " yes", " Yes"]]
        self.no_ids = [tok.encode(w, add_special_tokens=False)[0] for w in ["no", "No", " no", " No"]]

    def _sync(self):
        if str(self.device).startswith("cuda"):
            for d in (range(torch.cuda.device_count()) if self.multi else [self.device]):
                torch.cuda.synchronize(d)

    def _inputs(self, images, texts, system=None, image_pos="first"):
        """image_pos: "first" (our layout), "last" (text, then image) or "none" (text only)."""
        sys_ = [{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []
        img = [] if image_pos == "none" else [{"type": "image"}]
        msgs = [sys_ + [{"role": "user", "content": ([{"type": "text", "text": t}] + img if image_pos != "first"
                                                    else img + [{"type": "text", "text": t}])}] for t in texts]
        prompts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in msgs]
        extra = {}
        if "internvl" in self.model_id.lower():
            extra["crop_to_patches"] = False  # one 448px tile = 256 image tokens (InternVLProcessor tiles by default)
        if image_pos == "none":
            return self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        return self.processor(text=prompts, images=images, return_tensors="pt", padding=True, **extra).to(self.device)

    @torch.inference_mode()
    def generate(self, images, texts, max_new_tokens=32, sample=None, n=1, hidden_layers=(),
                 want_logits=True):
        """sample=None -> greedy; sample=dict(temperature=..,top_p=..,top_k=..) or 'model' (generation_config)."""
        inputs = self._inputs(images, texts)
        captured, hooks = {}, []
        for li in hidden_layers:
            def hook(_m, _i, out, li=li):
                h = out[0] if isinstance(out, tuple) else out
                if h.shape[1] > 1 and li not in captured:  # prefill pass only
                    captured[li] = h[:, -1, :].float().cpu()
            hooks.append(self.layers[li].register_forward_hook(hook))
        if sample is None:
            kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        elif sample == "model":
            kw = dict(do_sample=True)
        else:
            kw = dict(do_sample=True, **sample)
        self._sync()
        t0 = time.time()
        try:
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=n,
                                      repetition_penalty=1.0, return_dict_in_generate=True,
                                      output_logits=want_logits, **kw)
        finally:
            for h in hooks:
                h.remove()
        self._sync()
        dt = time.time() - t0
        plen = inputs["input_ids"].shape[1]
        seqs = out.sequences[:, plen:]
        n_prompt = inputs["attention_mask"].sum(1).repeat_interleave(n).tolist()
        results = []
        if want_logits:
            logp = torch.log_softmax(torch.stack(out.logits, 1).float(), -1)  # (B*n, T, V)
            seqs = seqs.to(logp.device)
        for i in range(seqs.shape[0]):
            ids = seqs[i].tolist()
            L = next((j for j, t in enumerate(ids) if t in self.eos_ids), len(ids))
            r = dict(text=self.processor.tokenizer.decode(ids[:L], skip_special_tokens=True).strip(),
                     n_prompt=int(n_prompt[i]), n_gen=L + (L < len(ids)), truncated=L == len(ids))
            if want_logits:
                lp = logp[i, :max(L, 1)]
                chosen = lp.gather(-1, seqs[i, :max(L, 1), None]).squeeze(-1)
                r["tok_logprobs"] = [round(x, 5) for x in chosen.tolist()]
                r["tok_entropy"] = [round(x, 5) for x in (-(lp.exp() * lp).sum(-1)).tolist()]
            results.append(r)
        tot = sum(r["n_prompt"] + r["n_gen"] for r in results)
        for r in results:  # attribute batch wall time by token share (prefill dominates for short answers)
            r["gpu_s"] = round(dt * (r["n_prompt"] + r["n_gen"]) / tot, 5)
        hidden = {li: captured[li] for li in captured}
        return results, hidden

    @torch.inference_mode()
    def p_yes(self, images, texts, system=None):
        """P(yes) / (P(yes) + P(no)) at the first generated position -- the P(True) feature. `yes_raw` is the
        unnormalised probability of the yes tokens (the quantity a critic model thresholds)."""
        inputs = self._inputs(images, texts, system)
        self._sync()
        t0 = time.time()
        logits = self.model(**inputs).logits[:, -1, :].float()
        self._sync()
        dt = time.time() - t0
        p = torch.softmax(logits, -1)
        y, no = p[:, self.yes_ids].sum(-1), p[:, self.no_ids].sum(-1)
        n_prompt = inputs["attention_mask"].sum(1).tolist()
        tot = sum(n_prompt)
        return [dict(p_yes=round(float(a / (a + b + 1e-12)), 5), yes_raw=round(float(a), 5), n_prompt=int(n), n_gen=1,
                     gpu_s=round(dt * n / tot, 5)) for a, b, n in zip(y, no, n_prompt)]

    def word_ids(self, words):
        """First token of each word as it can open a reply (lower / capitalised, with and without a leading space)."""
        tok = self.processor.tokenizer
        return sorted({tok.encode(v, add_special_tokens=False)[0] for w in words for v in (w, w.capitalize(), " " + w, " " + w.capitalize())})

    @torch.inference_mode()
    def first_token(self, images, texts, retrieve, skip, image_pos="first", max_new_tokens=6):
        """A published retrieve / skip prompt: greedy reply of a few tokens, and the probability mass of the two word
        groups at the first generated token that contains a letter (a reply may open with a quotation mark):
        p_retrieve = P(retrieve words) / (P(retrieve words) + P(skip words))."""
        inputs = self._inputs(images, texts, image_pos=image_pos)
        self._sync()
        t0 = time.time()
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                                  repetition_penalty=1.0, return_dict_in_generate=True, output_logits=True)
        self._sync()
        dt = time.time() - t0
        tok = self.processor.tokenizer
        seqs = out.sequences[:, inputs["input_ids"].shape[1]:]
        P = torch.softmax(torch.stack(out.logits, 1).float(), -1)  # (B, T, V)
        ra, sk = self.word_ids(retrieve), self.word_ids(skip)
        n_prompt = inputs["attention_mask"].sum(1).tolist()
        tot = sum(n_prompt)
        rows = []
        for i in range(seqs.shape[0]):
            ids = seqs[i].tolist()
            L = next((j for j, t in enumerate(ids) if t in self.eos_ids), len(ids))
            k = next((j for j in range(L) if any(ch.isalpha() for ch in tok.decode([ids[j]]))), 0)
            x, y = float(P[i, k, ra].sum()), float(P[i, k, sk].sum())
            rows.append(dict(text=tok.decode(ids[:L], skip_special_tokens=True).strip(), p_retrieve=round(x / (x + y + 1e-12), 5),
                             retrieve_raw=round(x, 5), skip_raw=round(y, 5), step=k, n_prompt=int(n_prompt[i]), n_gen=L + (L < len(ids)),
                             gpu_s=round(dt * n_prompt[i] / tot, 5)))
        return rows

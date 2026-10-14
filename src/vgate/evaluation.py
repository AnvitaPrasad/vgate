"""Per-example correctness for InfoSeek and Encyclopedic-VQA (official protocols).

Used to build gate labels  u = correct(with evidence) - correct(no evidence).
Deterministic: EM is pure string ops; BEM runs in eval mode under no_grad (fp32).
Raw generations can be passed everywhere; normalize_prediction() is applied internally.
"""
from __future__ import annotations

import functools
import re
import string
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
INFOSEEK_PARQUET = _ROOT / "data/raw/m2kr/Infoseek_data/test-00000-of-00001.parquet"
EVQA_PARQUET = _ROOT / "data/raw/evqa_preflmr/data/test-00000-of-00001.parquet"
BEM_MODEL = "kortukov/answer-equivalence-bem"


# ----------------------------------------------------------------------------- shared
def normalize_prediction(text: str | None) -> str:
    """Strip reasoning / formatting noise from a raw generation before scoring."""
    text = text or ""
    if "</think>" in text:                     # drop everything up to and incl. the last </think>
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:                    # unterminated thinking block -> no answer
        return ""
    text = re.sub(r"^\s*answer\s*:\s*", "", text, flags=re.I)
    text = text.strip().split("\n", 1)[0]      # first line only
    return text.strip().rstrip(".").strip()


# ----------------------------------------------------------------------------- InfoSeek
# Per-example logic vendored from the official scorer (edchengg/infoseek_eval @ f43d0c6,
# infoseek_eval.py), trimmed but otherwise unchanged. Original notice:
#   MIT License. Copyright (c) 2023 edchengg
#   Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#   and associated documentation files (the "Software"), to deal in the Software without
#   restriction, including without limitation the rights to use, copy, modify, merge, publish,
#   distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#   The above copyright notice and this permission notice shall be included in all copies or
#   substantial portions of the Software.
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
#   BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#   DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """lowercase -> drop punctuation -> drop articles -> collapse whitespace (SQuAD style)."""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def clean_str_range(text: str) -> str:
    """'9-10' -> '9 - 10' so the minus is not read as a sign."""
    return "".join(" - " if (c == "-" and i >= 1 and text[i - 1].isdigit()) else c
                   for i, c in enumerate(text))


def process_numerical_answer(string_number: str) -> float | list[float]:
    """Parse a prediction into a number or [lo, hi] (first two numbers); [0, 0] if none."""
    nums = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",
                      clean_str_range(string_number))
    nums = [n.replace(",", "").strip(".") for n in nums]
    nums = [float(n.split(".")[0] if n.count(".") > 1 else n) for n in nums][:2]
    if len(nums) == 2:
        return nums if nums[0] <= nums[1] else nums[0]   # official: descending pair -> first value
    return nums[0] if nums else [0, 0]


def range_intersection_over_union(x: list[float], y: list[float]) -> float:
    overlap = max(0.0, min(max(x), max(y)) - max(min(x), min(y)))
    union = (max(x) - min(x) + 1e-12) + (max(y) - min(y) + 1e-12) - overlap
    return overlap / union if union != 0 else 0


def metric_numerical_range(pred, answer, tolerance: float = 0.1) -> int:
    """Scalar answer -> +/-10% range. Scalar pred: in-range. Range pred: contained or IoU>=0.5."""
    answer = list(answer) if isinstance(answer, tuple) else answer
    pred = list(pred) if isinstance(pred, tuple) else pred
    if not isinstance(answer, list):
        answer = [answer * (1 - tolerance), answer * (1 + tolerance)]
    if not isinstance(pred, list):
        return int(answer[0] <= pred <= answer[1])
    if answer[0] <= pred[0] <= answer[1] and answer[0] <= pred[1] <= answer[1]:
        return 1
    return int(range_intersection_over_union(pred, answer) >= 0.5 - 1e-12)


def harmonic_mean(*args: float) -> float:
    args = [a if a != 0 else 1e-12 for a in args]
    return len(args) / sum(1.0 / a for a in args)


def _numerical_reference(ref: dict) -> float | list[float]:
    """Official prepare_qid2example turns answer_eval=[{'wikidata': v, 'range': [lo, hi]}] into
    [lo, hi]. M2KR flattens that dict: Numerical rows have answer_eval=[] and the (already +/-10%
    or wikidata-bounded) range in `wikidata_range`. A lone value falls back to +/-10% tolerance."""
    rng, ae = ref.get("wikidata_range"), ref.get("answer_eval")
    if rng is None or len(rng) == 0:           # raw official format
        if isinstance(ae, dict):
            rng = ae.get("range")
        elif ae is not None and len(ae) and isinstance(ae[0], dict):
            rng = ae[0].get("range")
    if rng is not None and len(rng) >= 2:
        return [float(a) for a in list(rng)[:2]]
    if rng is not None and len(rng) == 1:
        return float(rng[0])
    return float(ref["wikidata_value"])


def infoseek_correct(pred: str, ref: dict) -> int:
    """ref: row with question_type, answer_eval, wikidata_value, wikidata_range (M2KR schema)."""
    pred = normalize_prediction(pred)
    qtype = ref["question_type"].lower()
    if qtype == "numerical":
        return metric_numerical_range(process_numerical_answer(pred), _numerical_reference(ref))
    # 'string' and 'time' are both max-EM over answer_eval in the official code
    return int(max(exact_match_score(pred, gt) for gt in ref["answer_eval"]))


def _infoseek_summary(rows: list[tuple[str, int]]) -> dict:
    """Official evaluate_infoseek(): percentages rounded to 2 dp (+ n)."""
    def pct(xs):
        return round(100 * sum(xs) / len(xs), 2) if xs else 0.0
    by = {t: [c for q, c in rows if q == t] for t in ("time", "numerical", "string")}
    return {"score": pct([c for _, c in rows]), "score_time": pct(by["time"]),
            "score_num": pct(by["numerical"]), "score_string": pct(by["string"]), "n": len(rows)}


def infoseek_score(preds: dict[str, str], refs: dict[str, dict]) -> dict:
    """Like the official evaluate(): only qids present in both are scored; final = hmean of the
    unseen_question / unseen_entity split scores (a missing split therefore drives it to ~0)."""
    rows = {"unseen_question": [], "unseen_entity": []}
    for qid in sorted(set(preds) & set(refs)):
        ref = refs[qid]
        split = "unseen_question" if ref["data_split"].endswith("unseen_question") else "unseen_entity"
        qtype = ref["question_type"].lower()
        rows[split].append((qtype if qtype in ("time", "numerical") else "string",
                            infoseek_correct(preds[qid], ref)))
    uq, ue = _infoseek_summary(rows["unseen_question"]), _infoseek_summary(rows["unseen_entity"])
    return {"final_score": round(harmonic_mean(uq["score"], ue["score"]), 2),
            "overall": _infoseek_summary(rows["unseen_question"] + rows["unseen_entity"]),
            "unseen_question_score": uq, "unseen_entity_score": ue,
            "n_missing_preds": len(set(refs) - set(preds))}


def load_infoseek_refs(path=INFOSEEK_PARQUET) -> dict[str, dict]:
    """qid -> M2KR row with the fields infoseek_correct / infoseek_score need."""
    import pandas as pd
    cols = ["question_id", "question_type", "answer_eval", "wikidata_value", "wikidata_range",
            "data_split"]
    return {r["question_id"]: r for r in pd.read_parquet(path, columns=cols).to_dict("records")}


# ----------------------------------------------------------------------------- E-VQA
# EM normalisation from google-research/encyclopedic_vqa/evaluation_utils.py @ 5b09c22
# (Copyright The Google Research Authors, Apache-2.0). _CONTRACTIONS is omitted: it runs after
# punctuation removal, so it only rewrites apostrophe-free keys to distinct apostrophe-bearing
# values (injective, verified) -> it can never change an EM decision.
_EVQA_PUNCT = set(string.punctuation + "‘’´`_")
_DIGIT_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
              "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
              "entailment": "yes", "true": "yes", "contradiction": "no", "false": "no"}


def preprocess_answer(answer: str) -> str:
    answer = answer.lower().replace("\n", " ").replace("\t", " ").strip()
    if answer.startswith("<extra_id_0> "):
        answer = answer.replace("<extra_id_0> ", "")
    answer = "".join("" if c in _EVQA_PUNCT else c for c in answer)
    answer = re.sub(r"\b(the answer is|a|an|the)\b", " ", answer)
    return " ".join(_DIGIT_MAP.get(w, w) for w in answer.split())


@functools.lru_cache(maxsize=None)
def _load_bem(model_name: str, device: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float32)
    return tok, model.to(device).eval()


class BEMScorer:
    """BEM answer-equivalence (Bulian et al. 2022), PyTorch port of the TF-Hub model.

    Input layout follows the original, `text.combine_segments((candidate, reference, question))`
    in evaluation_utils.py, i.e. [CLS] cand [SEP] ref [SEP] question [SEP], built exactly as in
    the kortukov model card (text="[CLS] c [SEP]", pair="r [SEP] q [SEP]", no auto specials).
    Returns P(equivalent) = softmax(logits)[:, 1]."""

    def __init__(self, device: str = "cpu", batch_size: int = 64, model_name: str = BEM_MODEL):
        self.device, self.batch_size = device, batch_size
        self.tok, self.model = _load_bem(model_name, device)

    def score(self, questions, references, candidates) -> np.ndarray:
        import torch
        assert len(questions) == len(references) == len(candidates)
        out = []
        for i in range(0, len(candidates), self.batch_size):
            sl = slice(i, i + self.batch_size)
            enc = self.tok(text=[f"[CLS] {c} [SEP]" for c in candidates[sl]],
                           text_pair=[f"{r} [SEP] {q} [SEP]"
                                      for r, q in zip(references[sl], questions[sl])],
                           add_special_tokens=False, padding=True, truncation=True,
                           max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits.float()
            out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def evqa_correct_batch(preds, questions, answers_list, scorer: BEMScorer | None = None,
                       threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Official single-answer protocol, max over references: EM(preprocess_answer) -> 1, else
    BEM >= 0.5. One batched BEM pass over all (pred, ref) pairs that missed EM.
    Returns (correct int array, score float array; score = 1.0 where EM short-circuits)."""
    n = len(preds)
    correct, score = np.zeros(n, dtype=int), np.zeros(n, dtype=np.float32)
    idx, qs, rs, cs = [], [], [], []
    for i, (pred, q, answers) in enumerate(zip(preds, questions, answers_list)):
        cand = normalize_prediction(pred)
        refs = list(dict.fromkeys(a for a in answers if preprocess_answer(a)))  # dedup, keep order
        if not refs:
            raise ValueError(f"row {i}: reference answers empty after preprocessing")
        if any(preprocess_answer(cand) == preprocess_answer(r) for r in refs):
            correct[i], score[i] = 1, 1.0
            continue
        idx += [i] * len(refs); qs += [q] * len(refs); rs += refs; cs += [cand] * len(refs)
    if idx:
        bem = (scorer or BEMScorer()).score(qs, rs, cs)  # model load is cached
        np.maximum.at(score, np.array(idx), bem)
        miss = np.array(sorted(set(idx)))
        correct[miss] = (score[miss] >= threshold).astype(int)
    return correct, score


def evqa_correct(pred: str, question: str, answers, scorer: BEMScorer | None = None
                 ) -> tuple[int, float]:
    c, s = evqa_correct_batch([pred], [question], [list(answers)], scorer)
    return int(c[0]), float(s[0])


# ----------------------------------------------------------------------------- OK-VQA
def okvqa_soft_acc(pred: str, answers) -> float:
    """Official VQA accuracy: mean over the 10 leave-one-annotator-out subsets of min(1, #matches / 3),
    with VQA answer normalisation (preprocess_answer)."""
    p = preprocess_answer(normalize_prediction(pred))
    gts = [preprocess_answer(a) for a in answers]
    accs = [min(1.0, sum(g == p for j, g in enumerate(gts) if j != i) / 3.0) for i in range(len(gts))]
    return float(sum(accs) / len(accs)) if accs else 0.0


def okvqa_correct(pred: str, answers, threshold: float = 0.6) -> tuple[int, float]:
    """Binary label fixed in PREREG v2: correct iff soft VQA accuracy >= 0.6 (>= 2 annotators agree)."""
    s = okvqa_soft_acc(pred, answers)
    return int(s >= threshold - 1e-9), s


def okvqa_relaxed(pred: str, answers, correct: int) -> int:
    """Containment-relaxed SENSITIVITY label (PREREG v2e, never a headline number): the official label, or an answer
    given by >= 2 annotators contains / is contained in the normalised prediction (both >= 3 characters). Exact-match
    VQA accuracy rejects answer-form changes such as "cake" -> "chocolate cake" or "double" -> "doubles"."""
    if correct:
        return 1
    p = preprocess_answer(normalize_prediction(pred)).replace("-", " ")
    gts = [preprocess_answer(a).replace("-", " ") for a in answers]
    common = {g for g in gts if gts.count(g) >= 2 and len(g) >= 3}
    return int(len(p) >= 3 and any(g in p or p in g for g in common))


def mc_letter(pred: str, options=None) -> str:
    """Option letter chosen by a multiple-choice answer ("B", "B.", "(B) a hotel", or the option text itself); "" if none."""
    t = (pred or "").strip()
    m = re.match(r"^\W*([A-D])(?:[\s\.\):,]|$)", t)
    for l, o in zip("ABCD", options or []):  # the option text alone ("A hotel" must not be read as the letter A)
        if normalize_answer(t) == normalize_answer(o):
            return l
    return m.group(1) if m else ""


def mc_correct(pred: str, answer: str, options=None) -> int:
    return int(mc_letter(pred, options) == answer)

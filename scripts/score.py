"""Score every greedy condition of a run with the official metrics -> scores__<dataset>__<model>.csv.

  python scripts/score.py --run experiments/runs/J0 --dataset evqa --model Qwen3-VL-4B-Instruct
Columns: qid, c_<cond>, bem_<cond> (E-VQA only) for cond in ne, ev, oracle, para, think (when present).
"""
import argparse
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from vgate.evaluation import (BEMScorer, evqa_correct_batch, infoseek_correct, load_infoseek_refs, mc_correct, mc_letter, okvqa_correct,  # noqa: E402
                              okvqa_relaxed)

GREEDY = ["ne", "ev", "oracle", "para", "think", "think_s2", "ne_tset", "ne_rerun", "ev_rerun",  # think*, ne_tset are sampled
          "ev1_1", "ev1_2", "ev1_3", "ev1_4", "ev1_5", "ev_keep"]  # PREREG v3: one passage at a time; keep-your-answer prompt
NUMW = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12"}


def numbers(text):
    """Numbers mentioned in a string (digits, ordinals like 10th, number words up to twelve)."""
    import re
    t = re.sub(r"\b(" + "|".join(NUMW) + r")\b", lambda m: NUMW[m.group(1)], str(text).lower())
    return {n.lstrip("0") or "0" for n in re.findall(r"\d+", t.replace(",", ""))}


def numeric_strict(correct, pred, answers):
    """Sensitivity label (not the official metric): if any reference contains a number, the prediction must
    share a number with some reference in addition to being judged correct -- BEM is lenient on numeric
    near-misses (audit: 9/50 E-VQA harm cases were BEM false positives such as 20 vs 25, 2007 vs 2008)."""
    ref_nums = [numbers(a) for a in answers if numbers(a)]
    if not correct or not ref_nums:
        return int(correct)
    return int(any(numbers(pred) & r for r in ref_nums))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    run = pathlib.Path(a.run)
    recs = {r["qid"]: r for r in map(json.loads, open(pathlib.Path(a.data) / "processed" / f"{a.dataset}.jsonl"))}
    out = {}
    scorer = BEMScorer() if a.dataset == "evqa" else None
    refs = load_infoseek_refs() if a.dataset == "infoseek" else None
    for cond in GREEDY:
        p = run / f"{a.dataset}__{a.model}__{cond}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p)]
        qids = [r["qid"] for r in rows]
        preds = [r["text"] for r in rows]
        if a.dataset == "evqa":
            cs, ss = evqa_correct_batch(preds, [recs[q]["question"] for q in qids],
                                        [recs[q]["answers"] for q in qids], scorer=scorer)
            for q, c, s, pr in zip(qids, cs, ss, preds):
                out.setdefault(q, {})[f"c_{cond}"], out[q][f"bem_{cond}"] = int(c), round(float(s), 4)
                out[q][f"cs_{cond}"] = numeric_strict(int(c), pr, recs[q]["answers"])
        elif a.dataset == "okvqa":
            for q, pr in zip(qids, preds):
                c, s = okvqa_correct(pr, recs[q]["answers"])
                out.setdefault(q, {})[f"c_{cond}"], out[q][f"soft_{cond}"] = c, round(s, 4)
                out[q][f"cr_{cond}"] = okvqa_relaxed(pr, recs[q]["answers"], c)  # sensitivity label, see evaluation.py
        elif a.dataset == "cvqa":  # multiple choice (PREREG v7): the letter must match
            for q, pr in zip(qids, preds):
                out.setdefault(q, {})[f"c_{cond}"] = mc_correct(pr, recs[q]["answers"][0], recs[q]["options"])
                out[q][f"letter_{cond}"] = mc_letter(pr, recs[q]["options"])
        else:
            for q, pr in zip(qids, preds):
                out.setdefault(q, {})[f"c_{cond}"] = infoseek_correct(pr, refs[q])
        print(f"{cond}: n={len(rows)} acc={sum(out[q][f'c_{cond}'] for q in qids) / len(rows):.4f}", flush=True)
    df = pd.DataFrame.from_dict(out, orient="index")
    df.index.name = "qid"
    df.to_csv(run / f"scores__{a.dataset}__{a.model}.csv")


if __name__ == "__main__":
    main()

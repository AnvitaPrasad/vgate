"""Where is the value of retrieval lost? (PREREG v3a; descriptive, no fitted parameter.)

Gold labels (is the gold item retrieved, at which rank) are used for ANALYSIS only and never as a gate feature.

  python scripts/loss_decomposition.py --split confirm   -> experiments/results/loss_decomposition__confirm.{json,md}
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
from vgate import gate  # noqa: E402

R, DATA = ROOT / "experiments/runs", ROOT / "data"
CAP = 160  # per-passage word cap of the top-5 prompt (prompts.context_block)
# model -> run per split
RUNS = {"Qwen3-VL-4B-Instruct": dict(dev="J1", confirm="J4"), "Qwen2.5-VL-3B-Instruct": dict(dev="J2", confirm="J2"),
        "InternVL3_5-4B-HF": dict(dev="J5", confirm="J5"), "Qwen3-VL-8B-Instruct": dict(dev="J7", confirm="J7")}


def gold_info(dataset):
    """qid -> (rank of the gold item among the five passages or 0, word count of that passage, answer string in the chunks)."""
    recs = [json.loads(l) for l in open(DATA / "processed" / f"{dataset}.jsonl")]
    raw = pd.read_parquet(DATA / "raw" / ("evqa_preflmr/data/test-00000-of-00001.parquet" if dataset == "evqa"
                                           else "infoseek_evaclip/data/validation-00000-of-00001.parquet"))
    gold = dict(zip(raw.question_id.astype(str), [set(map(str, g)) for g in raw.pos_item_ids]))
    out = {}
    for r in recs:
        pos = [i for i, e in enumerate(r["evidence"]) if e["id"] in gold[str(r["qid"])]]
        text = " ".join(e["text"] for e in r["evidence"]).lower()
        out[r["qid"]] = dict(rank=pos[0] + 1 if pos else 0, words=len(r["evidence"][pos[0]]["text"].split()) if pos else 0,
                             ans_in_chunks=int(any(a.lower() in text for a in r["answers"] if len(a) > 1)))
    return pd.DataFrame.from_dict(out, orient="index")


def ci(df, f, n_boot=2000):
    """95% image-cluster bootstrap of f(sub-frame) -> [point, lo, hi] in accuracy points."""
    if len(df) == 0:
        return None
    pt, lo, hi = gate.cluster_bootstrap(df.group.to_numpy(), lambda i: f(df.iloc[i]), n_boot=n_boot, alpha=0.05)
    return [round(100 * pt, 2), round(100 * lo, 2), round(100 * hi, 2)]


def mean(col, mask=None):
    def f(d):
        d = d if mask is None else d[mask(d)]
        return float(d[col].mean()) if len(d) else np.nan
    return f


def diff(col_a, mask_a, col_b, mask_b):
    return lambda d: mean(col_a, mask_a)(d) - mean(col_b, mask_b)(d)


def decompose(dataset, model, split, G):
    df = A.load(R / RUNS[model][split], DATA, dataset, model, split).join(G)
    has_or = "c_oracle" in df and df.c_oracle.notna().any()
    df["ideal_gate"] = np.where(df.lab_gold_in_top5 == 1, df.c_ev, df.c_ne)
    df["oracle_choice"] = np.maximum(df.c_ne, df.c_ev)
    in5 = lambda d: d["rank"] > 0
    o = dict(n=len(df), never=ci(df, mean("c_ne")), always=ci(df, mean("c_ev")), oracle_choice=ci(df, mean("oracle_choice")),
             ideal_gate=ci(df, mean("ideal_gate")), ideal_gate_minus_always=ci(df, diff("ideal_gate", None, "c_ev", None)),
             gold_top1=round(100 * df.lab_gold_in_top1.mean(), 1), gold_top5=round(100 * df.lab_gold_in_top5.mean(), 1),
             all_five_given_gold=ci(df, mean("c_ev", in5)))
    if has_or:
        o.update(gold_alone_given_gold=ci(df, mean("c_oracle", in5)), gold_alone_all=ci(df, mean("c_oracle")),
                 gold_alone_minus_all_five=ci(df, diff("c_oracle", in5, "c_ev", in5)))
        sel = np.where(df["rank"] > 0, df.c_oracle, df.c_ne)  # ideal passage selector: gold passage alone if retrieved
        o["ideal_selector"] = ci(df.assign(sel=sel), mean("sel"))
    if dataset == "evqa":
        full = lambda d: (d["rank"] > 0) & (d.words <= CAP)
        r1, r25 = (lambda d: full(d) & (d["rank"] == 1)), (lambda d: full(d) & (d["rank"] > 1))
        o.update(n_untruncated=int(full(df).sum()), n_rank1=int(r1(df).sum()), n_rank25=int(r25(df).sum()),
                 rank1=ci(df, mean("c_ev", r1)), rank25=ci(df, mean("c_ev", r25)), rank1_minus_rank25=ci(df, diff("c_ev", r1, "c_ev", r25)),
                 truncated=ci(df, mean("c_ev", lambda d: (d["rank"] > 0) & (d.words > CAP))), untruncated=ci(df, mean("c_ev", full)))
        for name, cb in (("cb_wrong", 0), ("cb_right", 1)):
            o[f"rank1_minus_rank25__{name}"] = ci(df, diff("c_ev", lambda d: r1(d) & (d.c_ne == cb), "c_ev", lambda d: r25(d) & (d.c_ne == cb)))
        if has_or:
            o.update(gold_alone_rank1=ci(df, mean("c_oracle", r1)), gold_alone_rank25=ci(df, mean("c_oracle", r25)),
                     gold_alone_minus_all_five_rank25=ci(df, diff("c_oracle", r25, "c_ev", r25)),
                     # added after the held-out run (claim check, 2026-10-05): the single-passage prompt shows up to 400
                     # words and the five-passage prompt 160 per passage, so the like-for-like gap is the untruncated one
                     gold_alone_untruncated=ci(df, mean("c_oracle", full)),
                     gold_alone_minus_all_five_untruncated=ci(df, diff("c_oracle", full, "c_ev", full)))
    if dataset == "infoseek":
        o.update(ans_in_chunks_given_gold=round(100 * df[in5(df)].ans_in_chunks.mean(), 1),
                 all_five_ans_in_chunks=ci(df, mean("c_ev", lambda d: in5(d) & (d.ans_in_chunks == 1))),
                 all_five_ans_missing=ci(df, mean("c_ev", lambda d: in5(d) & (d.ans_in_chunks == 0))))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="confirm")
    a = ap.parse_args()
    G = {d: gold_info(d) for d in ("evqa", "infoseek")}
    res = {m: {d: decompose(d, m, a.split, G[d]) for d in ("evqa", "infoseek")} for m in RUNS}
    f = lambda v: "n/a" if v is None else f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]"
    exp = {}
    V3A = ["Qwen3-VL-4B-Instruct", "Qwen2.5-VL-3B-Instruct"]  # the two models that had the gold-alone pass when v3a was registered
    full = [m for m in RUNS if "gold_alone_all" in res[m]["evqa"]]
    d3 = lambda m: res[m]["evqa"]["gold_alone_minus_all_five_rank25"][0] >= 10 and res[m]["evqa"]["gold_alone_minus_all_five_rank25"][1] > 0
    exp["D1"] = all(abs(res[m][d]["ideal_gate_minus_always"][0]) <= 2.0 for m in V3A for d in ("evqa", "infoseek"))
    exp["D2"] = all(res[m]["evqa"]["rank1_minus_rank25"][0] >= 10 and res[m]["evqa"]["rank1_minus_rank25"][1] > 0 for m in RUNS)
    exp["D3"] = all(d3(m) for m in V3A)
    for m in full:  # PREREG v4c registers D3 for models 3 and 4 once their gold-alone pass exists
        if m not in V3A:
            exp[f"D3 (v4c) {m}"] = bool(d3(m))
    lines = [f"# Loss decomposition ({a.split} split; PREREG v3a; accuracy points, 95% image-cluster bootstrap)", ""]
    for m in RUNS:
        e, i = res[m]["evqa"], res[m]["infoseek"]
        lines += [f"## {m}", "", "| | E-VQA | InfoSeek |", "|---|---|---|"]
        for key, name in (("never", "never"), ("always", "always (top-5)"), ("oracle_choice", "oracle choice of the two answers"),
                          ("ideal_gate", "ideal read/skip gate (read iff gold retrieved)"), ("ideal_gate_minus_always", "ideal gate − always"),
                          ("all_five_given_gold", "gold in top-5: reading all five"), ("gold_alone_given_gold", "gold in top-5: gold passage alone"),
                          ("gold_alone_minus_all_five", "gold alone − all five"), ("ideal_selector", "ideal passage selector"),
                          ("gold_alone_all", "gold evidence for every question")):
            if key in e:
                lines.append(f"| {name} | {f(e[key])} | {f(i[key])} |")
        lines += [f"| gold in top-1 / top-5 (%) | {e['gold_top1']} / {e['gold_top5']} | {i['gold_top1']} / {i['gold_top5']} |", "",
                  f"E-VQA, gold in top-5 and untruncated (n={e['n_untruncated']}): gold at rank 1 (n={e['n_rank1']}) {f(e['rank1'])}; "
                  f"at ranks 2–5 (n={e['n_rank25']}) {f(e['rank25'])}; difference {f(e['rank1_minus_rank25'])} "
                  f"(closed-book wrong: {f(e['rank1_minus_rank25__cb_wrong'])}; closed-book right: {f(e['rank1_minus_rank25__cb_right'])})."]
        if "gold_alone_rank25" in e:
            lines.append(f"Gold passage alone: rank 1 {f(e['gold_alone_rank1'])}; ranks 2–5 {f(e['gold_alone_rank25'])}; "
                         f"gold alone − all five at ranks 2–5: {f(e['gold_alone_minus_all_five_rank25'])}.")
        if "gold_alone_minus_all_five_untruncated" in e:
            lines.append(f"Same passage text in both prompts (untruncated; added after the held-out run): gold alone "
                         f"{f(e['gold_alone_untruncated'])}, gold alone − all five {f(e['gold_alone_minus_all_five_untruncated'])}.")
        lines += [f"E-VQA, gold in top-5: untruncated {f(e['untruncated'])} vs truncated at {CAP} words {f(e['truncated'])}.",
                  f"InfoSeek, right page in top-5: an answer string is inside the five chunks in {i['ans_in_chunks_given_gold']}% of cases; "
                  f"accuracy {f(i['all_five_ans_in_chunks'])} with it, {f(i['all_five_ans_missing'])} without.", ""]
    lines += ["## Expectations (PREREG v3a)", ""] + [f"- {k}: {'held' if v else 'NOT held'}" for k, v in exp.items()]
    out = ROOT / "experiments/results" / f"loss_decomposition__{a.split}"
    out.with_suffix(".json").write_text(json.dumps(dict(split=a.split, results=res, expectations=exp), indent=1))
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

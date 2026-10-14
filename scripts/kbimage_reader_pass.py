"""PREREG v6c (PILOT, E-VQA development split): do the readers answer better with the image-re-ranked evidence?

Compares, paired by question, each reader's answers on the five passages that the image re-ranking of PreFLMR's top-20 puts
first (J19) with its answers on the retriever's own top-5 and its closed-book answers (J1 for Qwen3-VL-4B, J2 for
Qwen2.5-VL-3B). All three are scored here with the benchmark's metric (BEM), in one pass.

  python scripts/kbimage_reader_pass.py   -> experiments/results/kbimage_reader_pass.{json,md} and kbimage_reader_pass__questions.csv

E1: the main reader's always-retrieve accuracy rises by at least 3.0 points, 80% image-cluster interval above zero.
E2: the second reader's rises by at least 2.0 points.  E3: the main reader's harm rate does not rise by more than 1.0 point.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vgate import gate  # noqa: E402
from vgate.evaluation import BEMScorer, evqa_correct_batch  # noqa: E402

R = ROOT / "experiments/runs"
READERS = dict(dev={"Qwen3-VL-4B-Instruct": "J1", "Qwen2.5-VL-3B-Instruct": "J2"},      # reader -> run with its closed-book and top-5 answers
               confirm={"Qwen3-VL-4B-Instruct": "J4", "Qwen2.5-VL-3B-Instruct": "J2"})
SETUP = dict(dev=dict(run="J19", alpha=0.2, tag="", data="kaggle/_dataset_vgate-evqa-rerank", title="PREREG v6c — readers on the image-re-ranked evidence (pilot; E-VQA development split"),
             confirm=dict(run="J20", alpha=0.05, tag="__confirm", data="kaggle/_dataset_vgate-evqa-rerank-confirm",
                          title="PREREG v6d — readers on the image-re-ranked evidence (preregistered; E-VQA held-out split"))


def answers(path, qids):
    rows = {r["qid"]: r["text"] for r in map(json.loads, open(path))}
    return [rows[q] for q in qids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "confirm"])
    a = ap.parse_args()
    S, readers = SETUP[a.split], READERS[a.split]
    dev = (ROOT / f"data/splits/evqa_{a.split}.txt").read_text().split()
    recs = {r["qid"]: r for r in map(json.loads, open(ROOT / "data/processed/evqa.jsonl")) if r["qid"] in set(dev)}
    rr = {r["qid"]: r for r in map(json.loads, open(ROOT / S["data"] / "processed/evqa.jsonl"))}
    qids = sorted(recs)
    g = np.array([recs[q]["image_id"] for q in qids])
    src = np.array(["landmarks" if "landmark" in recs[q]["image_file"] else "inaturalist" for q in qids])
    gold0 = np.array([bool(recs[q]["lab_gold_in_top5"]) for q in qids]); gold1 = np.array([bool(rr[q]["lab_gold_in_top5"]) for q in qids])
    scorer = BEMScorer()
    Q, Aref = [recs[q]["question"] for q in qids], [recs[q]["answers"] for q in qids]
    rep, L = dict(n=len(qids), gold_passage_among_five=dict(retriever=round(100 * float(gold0.mean()), 1), reranked=round(100 * float(gold1.mean()), 1))), []
    ci = lambda d, k=None: [round(100 * x, 2) for x in gate.cluster_bootstrap(g if k is None else g[k], (lambda i: float(d[i].mean())) if k is None else (lambda i: float(d[k][i].mean())),
                                                                              n_boot=2000, alpha=S["alpha"])]
    per_q = []  # one row per reader and question, for analyses that need the paired outcomes (scripts/rerank_gate_pilot.py)
    for model, run in readers.items():
        f = R / S["run"] / f"evqa__{model}__ev.jsonl"
        if not f.exists() or sum(1 for _ in open(f)) < len(qids):
            rep[model] = f"NOT_RUN: no complete {S['run']} output for this reader"
            continue
        c = {}
        for name, path in (("ne", R / run / f"evqa__{model}__ne.jsonl"), ("ev", R / run / f"evqa__{model}__ev.jsonl"), ("rr", f)):
            c[name] = np.array(evqa_correct_batch(answers(path, qids), Q, Aref, scorer=scorer)[0], float)
        d = c["rr"] - c["ev"]
        per_q.append(pd.DataFrame(dict(qid=qids, group=g, source=src, gold_retriever=gold0.astype(int), gold_reranked=gold1.astype(int), model=model,
                                       c_ne=c["ne"].astype(int), c_ev=c["ev"].astype(int), c_rr=c["rr"].astype(int))))
        u0, u1 = c["ev"] - c["ne"], c["rr"] - c["ne"]
        h0, m0, h1, m1 = float((u0 > 0).mean()), float((u0 < 0).mean()), float((u1 > 0).mean()), float((u1 < 0).mean())
        cell = dict(never=round(100 * float(c["ne"].mean()), 1), always_retriever=round(100 * float(c["ev"].mean()), 1), always_reranked=round(100 * float(c["rr"].mean()), 1),
                    diff=ci(d), help=dict(retriever=round(100 * h0, 1), reranked=round(100 * h1, 1)), harm=dict(retriever=round(100 * m0, 1), reranked=round(100 * m1, 1)),
                    harm_diff=ci((u1 < 0).astype(float) - (u0 < 0).astype(float)),
                    by_source={s: dict(n=int((src == s).sum()), diff=ci(d, src == s)) for s in ("inaturalist", "landmarks")},
                    where=dict(gold_gained=dict(n=int((gold1 & ~gold0).sum()), right_before=round(100 * float(c["ev"][gold1 & ~gold0].mean()), 1),
                                                right_after=round(100 * float(c["rr"][gold1 & ~gold0].mean()), 1)),
                               gold_lost=dict(n=int((gold0 & ~gold1).sum()), right_before=round(100 * float(c["ev"][gold0 & ~gold1].mean()), 1) if (gold0 & ~gold1).any() else None,
                                              right_after=round(100 * float(c["rr"][gold0 & ~gold1].mean()), 1) if (gold0 & ~gold1).any() else None),
                               gold_in_both=dict(n=int((gold0 & gold1).sum()), right_before=round(100 * float(c["ev"][gold0 & gold1].mean()), 1),
                                                 right_after=round(100 * float(c["rr"][gold0 & gold1].mean()), 1)),
                               gold_in_neither=dict(n=int((~gold0 & ~gold1).sum()), right_before=round(100 * float(c["ev"][~gold0 & ~gold1].mean()), 1),
                                                    right_after=round(100 * float(c["rr"][~gold0 & ~gold1].mean()), 1))),
                    planner=dict(retriever=dict(floor=round(gate.auroc_needed(h0, m0, shape="any"), 3), needed=round(gate.auroc_needed(h0, m0), 3)),
                                 reranked=dict(floor=round(gate.auroc_needed(h1, m1, shape="any"), 3), needed=round(gate.auroc_needed(h1, m1), 3))))
        rep[model] = cell
        L += [f"## {model}", "",
              f"Never-retrieve {cell['never']}%; always-retrieve with the retriever's top-5 {cell['always_retriever']}%, with the re-ranked five {cell['always_reranked']}%: "
              f"**{cell['diff'][0]:+.2f} [{cell['diff'][1]:+.2f}, {cell['diff'][2]:+.2f}]** ({100 - int(100 * S['alpha'])}% image-cluster interval). "
              f"Retrieval helps {cell['help']['retriever']}% -> {cell['help']['reranked']}% and harms {cell['harm']['retriever']}% -> {cell['harm']['reranked']}% of questions "
              f"(harm: {cell['harm_diff'][0]:+.2f} [{cell['harm_diff'][1]:+.2f}, {cell['harm_diff'][2]:+.2f}]).",
              "By photograph: " + "; ".join(f"{s} (n = {v['n']}) {v['diff'][0]:+.2f} [{v['diff'][1]:+.2f}, {v['diff'][2]:+.2f}]" for s, v in cell["by_source"].items()) + ".",
              "Where (share answered correctly with the retriever's five -> with the re-ranked five): " + "; ".join(
                  f"{k.replace('_', ' ')} (n = {v['n']}) {v['right_before']} -> {v['right_after']}" for k, v in cell["where"].items()) + ".",
              f"Planner: AUROC floor / needed for a one-point lead {cell['planner']['retriever']['floor']} / {cell['planner']['retriever']['needed']} with the retriever's five, "
              f"{cell['planner']['reranked']['floor']} / {cell['planner']['reranked']['needed']} with the re-ranked five.", ""]
    st = ROOT / S["data"] / "rerank_stats.json"  # written by kbimage_rerank_export.py next to the evidence (that folder is not tracked)
    if st.exists():
        rep["gold_page_among_five"] = json.loads(st.read_text())["gold_page_among_five"]
    m1_, m2_ = list(readers)
    if isinstance(rep.get(m1_), dict):
        rep["E1"] = bool(rep[m1_]["diff"][0] >= 3.0 and rep[m1_]["diff"][1] > 0)
        rep["E3"] = bool(rep[m1_]["harm_diff"][0] <= 1.0)
    if isinstance(rep.get(m2_), dict):
        rep["E2"] = bool(rep[m2_]["diff"][0] >= 2.0)
    head = [f"# {S['title']}, n = {len(qids)})", "",
            f"Gold passage among the five read: {rep['gold_passage_among_five']['retriever']}% with the retriever's order, {rep['gold_passage_among_five']['reranked']}% re-ranked.", "",
            "Expectations: " + "; ".join(f"{k} {'held' if rep[k] else 'NOT held'}" for k in ("E1", "E2", "E3") if k in rep) + ".", ""]
    out = ROOT / ("experiments/results/kbimage_reader_pass" + S["tag"])
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(head + L) + "\n")
    if per_q:
        pd.concat(per_q).to_csv(str(out) + "__questions.csv", index=False, lineterminator="\n")
    print("\n".join(head + L))


if __name__ == "__main__":
    main()

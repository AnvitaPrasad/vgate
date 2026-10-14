"""Gate features from run outputs + the per-pass cost ledger.

Every feature is computed from passes a deployed gate could actually run BEFORE reading evidence (NE greedy,
NE samples, entity samples, P(True), paraphrase) or, for post-search gates, from retriever scores only.
Nothing here may read `lab_*`, gold, oracle or evidence-conditioned outputs (see data.FEATURE_BLACKLIST).
"""
import json
import pathlib
import re
import string

import numpy as np
import pandas as pd

from .evaluation import normalize_prediction

# Which pass each feature group needs (for cost accounting of a policy that uses it).
FEATURE_PASSES = {
    "free": ["ne"],                      # greedy NE pass: logprobs/entropy come for free
    "sample": ["ne", "ne_sample"],
    "entity": ["ne", "ent", "ent_sample"],
    "ptrue": ["ne", "ptrue"],
    "para": ["ne", "para"],
    "qpp": ["ne"],                       # retriever scores: search call only, no extra VLM pass
}
FREE = ["ent_mean", "ent_max", "ent_first", "lp_min", "lp_mean", "seq_lp", "n_gen_ne"]
SAMPLE = ["sample_agree", "sample_distinct"]
ENTITY = ["ent_agree", "ent_distinct"]
PTRUE = ["p_true"]
PARA = ["para_agree"]
QPP = ["qpp_top1_z", "qpp_margin_z", "qpp_nqc", "qpp_entropy"]


def norm(a):
    a = normalize_prediction(a or "").lower()
    a = "".join(ch for ch in a if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", a).split())


def read(run_dir, dataset, model, cond):
    p = pathlib.Path(run_dir) / f"{dataset}__{model}__{cond}.jsonl"
    if not p.exists():
        return {}
    return {d["qid"]: d for d in map(json.loads, open(p))}


def qpp(scores):
    """Scale-free query-performance-prediction features from the retriever's top-20 scores."""
    s = np.asarray(scores, dtype=float)
    mu, sd = s.mean(), s.std() + 1e-9
    top = s[:10]
    z = (top - mu) / sd
    p = np.exp(z - z.max()); p /= p.sum()
    return dict(qpp_top1_z=float(z[0]), qpp_margin_z=float((s[0] - s[1]) / sd),
                qpp_nqc=float(top.std() / (abs(mu) + 1e-9)), qpp_entropy=float(-(p * np.log(p)).sum()))


def free_features(ne):
    """The FREE feature group from one closed-book (NE) generation record: token entropies / log-probs / length."""
    lp, en = np.array(ne["tok_logprobs"] or [0.0]), np.array(ne["tok_entropy"] or [0.0])
    return dict(ent_mean=en.mean(), ent_max=en.max(), ent_first=en[0], lp_min=lp.min(), lp_mean=lp.mean(),
                seq_lp=lp.sum(), n_gen_ne=ne["n_gen"])


def build(run_dir, dataset, model, recs):
    """-> DataFrame indexed by qid: features, NE/EV answers, and per-pass token + GPU-second costs."""
    P = {c: read(run_dir, dataset, model, c) for c in
         ["ne", "ne_sample", "ent", "ent_sample", "ptrue", "para", "ev", "oracle"]}
    rows = []
    for r in recs:
        q = r["qid"]
        ne = P["ne"].get(q)
        if ne is None:
            continue
        g = norm(ne["text"])
        row = dict(qid=q, image_id=r["image_id"], ne_text=ne["text"], **free_features(ne), **qpp(r["retr_scores"]))
        if q in P["ne_sample"]:
            s = [norm(x) for x in P["ne_sample"][q]["samples"]]
            row.update(sample_agree=np.mean([x == g for x in s]), sample_distinct=len(set(s + [g])) / (len(s) + 1))
        if q in P["ent"] and q in P["ent_sample"]:
            e0, es = norm(P["ent"][q]["text"]), [norm(x) for x in P["ent_sample"][q]["samples"]]
            row.update(ent_text=P["ent"][q]["text"], ent_agree=np.mean([x == e0 for x in es]),
                       ent_distinct=len(set(es + [e0])) / (len(es) + 1))
        if q in P["ptrue"]:
            row["p_true"] = P["ptrue"][q]["p_yes"]
        if q in P["para"]:
            row["para_agree"] = float(norm(P["para"][q]["text"]) == g)
        for c in ["ne", "ne_sample", "ent", "ent_sample", "ptrue", "para", "ev", "oracle"]:
            if q in P[c]:
                row[f"tok_{c}"] = P[c][q]["n_prompt"] + P[c][q]["n_gen"]
                row[f"gpu_{c}"] = P[c][q]["gpu_s"]
        for c in ["ev", "oracle"]:
            if q in P[c]:
                row[f"{c}_text"] = P[c][q]["text"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("qid")


def base_cost(df, groups, unit="tok"):
    """Per-query cost of the passes needed by a feature set (union of passes, each counted once)."""
    passes = sorted({p for g in groups for p in FEATURE_PASSES[g]})
    return sum(df[f"{unit}_{p}"].fillna(0) for p in passes)


def load_hidden(run_dir, dataset, model, cond="ne"):
    """{layer: DataFrame(qid x hidden)} from a pass's hook-captured last-prompt-token states (all parts).
    cond="ne" is the closed-book pass; cond="ev" the evidence pass (run.py --hidden_conds ev)."""
    run = pathlib.Path(run_dir)
    out = {}
    pre = "hidden" if cond == "ne" else f"hidden-{cond}"
    for qp in sorted(run.glob(f"{pre}_{dataset}__{model}__qids*.json")):
        part = qp.stem.rsplit("qids", 1)[1]
        qids = json.loads(qp.read_text())
        for hp in sorted(run.glob(f"{pre}_{dataset}__{model}__L*_part{part}.npy")):
            layer = int(hp.stem.split("__L")[1].split("_")[0])
            out.setdefault(layer, []).append(pd.DataFrame(np.load(hp).astype(np.float32), index=qids))
    return {l: pd.concat(v)[lambda d: ~d.index.duplicated()] for l, v in out.items()}

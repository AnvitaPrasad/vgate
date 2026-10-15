"""PREREG v11, step 2: the audit on the released logs of Chen et al. (arXiv:2502.18023), Mix set (600 questions, six sources).

Input: experiments/results/chen_logs__questions.csv (scripts/chen_logs_score.py: their graded token accuracy of each reader's
closed-book and search-augmented answer, and the decisions of their hard and soft knowledge-boundary models, a prompt baseline
and a human). Segment = the source of the question.

  python scripts/chen_logs_audit.py   -> experiments/results/chen_logs_audit.{json,md}
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vgate import gate  # noqa: E402

GATES = ["hard", "soft", "prompt", "human"]
NAMES = {"gpt4-o": "GPT-4o", "qwen-vl-max": "Qwen-VL-Max", "qwen-vl-2": "Qwen-VL-2", "qwen-vl-chat": "Qwen-VL-Chat", "deepseek-vl-chat": "DeepSeek-VL-Chat"}
# Mix, Acc., as printed in their Table 3 (Qwen-VL-Chat) and Table 11 (the others): never, always, prompt, hard, soft, human.
PRINTED = {"gpt4-o": dict(never=34.52, always=35.96, prompt=33.99, hard=35.90, soft=35.86, human=36.49),
           "qwen-vl-max": dict(never=32.35, always=34.78, prompt=33.00, hard=35.48, soft=34.84, human=35.51),
           "qwen-vl-2": dict(never=32.59, always=35.56, prompt=33.27, hard=36.29, soft=35.62, human=36.33),
           "qwen-vl-chat": dict(never=26.13, always=32.39, prompt=27.23, hard=32.73, soft=30.98, human=31.02),
           "deepseek-vl-chat": dict(never=25.81, always=35.23, prompt=26.55, hard=35.38, soft=33.06, human=32.73)}


def told_source(src, a0, a1, seed=0):
    """Read decisions of 'told the source': per source the better of never / always, chosen on four folds, applied to the fifth."""
    out = np.zeros(len(src), bool)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(src):
        for s in np.unique(src):
            k = tr[src[tr] == s]
            out[te[src[te] == s]] = a1[k].mean() > a0[k].mean()
    return out


# Mix, LLM-judge score (Qwen-Max), as printed in their Table 3 (Qwen-VL-Chat) and Table 4 (the others); PREREG v11b.
PRINTED_JUDGE = {"gpt4-o": dict(never=51.44, always=52.90, prompt=50.57, hard=54.10, soft=52.97, human=55.27),
                 "qwen-vl-max": dict(never=46.54, always=49.26, prompt=47.30, hard=50.64, soft=51.06, human=52.05),
                 "qwen-vl-2": dict(never=46.36, always=47.89, prompt=47.46, hard=49.31, soft=49.29, human=51.41),
                 "qwen-vl-chat": dict(never=34.44, always=38.60, prompt=34.98, hard=39.59, soft=39.93, human=38.29),
                 "deepseek-vl-chat": dict(never=34.96, always=45.18, prompt=35.71, hard=45.08, soft=43.35, human=42.20)}


def main(n_boot=2000):
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="token", choices=["token", "judge"], help="judge: PREREG v11b, scores of a judge model in --judge_run")
    ap.add_argument("--judge_run", default="experiments/runs/J21", help="folder with judge_part*.jsonl (reader, cond, row, score)")
    ap.add_argument("--tag", default="judge", help="suffix of the judge result files (PREREG v11c uses judge32)")
    ap.add_argument("--sources", default="", help="PREREG v14b: comma-separated sources to keep (results go to chen_logs_audit__<tag>)")
    a = ap.parse_args()
    judge = a.metric == "judge"
    d = pd.read_csv(ROOT / "experiments/results/chen_logs__questions.csv")
    d = d[d.set == "mix"]
    if a.sources:
        d = d[d.source.isin(a.sources.split(","))]; assert d.source.nunique() == len(a.sources.split(",")), "unknown source"
        d = d.sort_values(["reader", "row"]); d["row"] = d.groupby("reader").cumcount()
    printed = {} if a.sources else PRINTED_JUDGE if judge else PRINTED  # their printed numbers are for all six sources
    if judge:  # replace the two accuracy columns by the judge's scores, clipped to [0, 1]
        J = pd.DataFrame([json.loads(l) for f in sorted(glob.glob(str(ROOT / a.judge_run / "judge_part*.jsonl"))) for l in open(f)])
        J["score"] = J.score.clip(0, 1)
        w = J.pivot_table(index=["reader", "row"], columns="cond", values="score", aggfunc="first")
        assert len(w) == len(d) and w.notna().all().all(), "judge scores are incomplete"
        d = d.merge(w.reset_index(), on=["reader", "row"])
        d["acc_norag"], d["acc_rag"] = d["norag"], d["rag"]
    rep, L = dict(n=int(d.row.nunique()), sources=list(d.source.unique()), readers={}), []
    rng = np.random.default_rng(2027)
    idx = rng.integers(0, rep["n"], size=(n_boot, rep["n"]))
    for r, k in d.groupby("reader", sort=False):
        k = k.sort_values("row"); src, a0, a1 = k.source.to_numpy(), k.acc_norag.to_numpy(), k.acc_rag.to_numpy()
        acc = lambda read: np.where(read, a1, a0)
        P = dict(never=np.zeros(len(k), bool), always=np.ones(len(k), bool), **{g: k[g].to_numpy().astype(bool) for g in GATES})
        P["told the source"] = told_source(src, a0, a1)
        for g in ("hard", "soft"):  # the logged gate's own majority decision per source
            P[f"{g}: majority per source"] = np.array([k[g][k.source == s].mean() > 0.5 for s in src])
        P["oracle"] = a1 > a0
        cell = dict(policies={n: dict(accuracy=round(100 * float(acc(p).mean()), 2), search_rate=round(100 * float(p.mean()), 1)) for n, p in P.items()})
        told = acc(P["told the source"])
        for g in GATES + ["always", "hard: majority per source", "soft: majority per source"]:
            diff = acc(P[g]) - told
            b = diff[idx].mean(1)
            cell["policies"][g]["vs_told"] = [round(100 * float(diff.mean()), 2), round(100 * float(np.quantile(b, 0.025)), 2), round(100 * float(np.quantile(b, 0.975)), 2)]
        cell["told_in_sample"] = round(100 * float(np.mean([max(a0[src == s].mean(), a1[src == s].mean()) for s in rep["sources"]])), 2)
        cell["told_other_seeds"] = [round(100 * float(acc(told_source(src, a0, a1, seed=s_)).mean()), 2) for s_ in range(1, 6)]
        # robustness added after claim check 14: 200 other fold assignments of told-the-source
        rep_ = np.array([acc(told_source(src, a0, a1, seed=1000 + s_)).mean() for s_ in range(200)])
        cell["told_200_assignments"] = dict(mean=round(100 * float(rep_.mean()), 2), **{f"{g}_minus_told_mean": round(100 * float((acc(P[g]).mean() - rep_).mean()), 2) for g in ("hard", "soft")},
                                            **{f"{g}_ahead_share": round(100 * float((acc(P[g]).mean() > rep_).mean()), 1) for g in ("hard", "soft")})
        cell["hard_agrees_with_its_majority"] = round(100 * float((P["hard"] == P["hard: majority per source"]).mean()), 1)
        cell["by_source"] = {s: dict(never=round(100 * float(a0[src == s].mean()), 1), always=round(100 * float(a1[src == s].mean()), 1),
                                     hard_rate=round(100 * float(P["hard"][src == s].mean()), 0), soft_rate=round(100 * float(P["soft"][src == s].mean()), 0),
                                     told_rate=round(100 * float(P["told the source"][src == s].mean()), 0)) for s in rep["sources"]}
        c0, c1 = a0 > 0.5, a1 > 0.5  # their threshold for a correct answer (compute_upper_bound)
        h, m = float((c1 & ~c0).mean()), float((c0 & ~c1).mean())
        cell["planner"] = dict(help=round(100 * h, 1), harm=round(100 * m, 1), auroc_floor=round(gate.auroc_needed(h, m, shape="any"), 3), auroc_needed=round(gate.auroc_needed(h, m), 3),
                               by_source={s: dict(help=round(100 * float((c1 & ~c0)[src == s].mean()), 1), harm=round(100 * float((c0 & ~c1)[src == s].mean()), 1)) for s in rep["sources"]})
        if r in printed:
            cell["reproduction_max_abs_diff"] = round(max(abs(cell["policies"][n]["accuracy"] - v) for n, v in printed[r].items()), 2)
            cell["never_always_abs_diff"] = [round(abs(cell["policies"][n]["accuracy"] - printed[r][n]), 2) for n in ("never", "always")]
        rep["readers"][r] = cell
    R = rep["readers"]
    lead = lambda r, g: R[r]["policies"][g]["vs_told"]
    rep["K1"] = bool(all(lead(r, "hard")[0] <= 0 for r in ("gpt4-o", "qwen-vl-max")))
    rep["K2"] = bool(all(lead(r, "soft")[0] <= 0 for r in ("gpt4-o", "qwen-vl-max")))
    rep["K3"] = bool(R["gpt4-o"]["hard_agrees_with_its_majority"] >= 85.0)  # the decisions are the same file for every reader
    rep["K4"] = bool(sum(not (lead(r, "hard")[0] >= 1.0 and lead(r, "hard")[1] > 0) for r in R) >= 4)
    f = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
    L += ["# PREREG v11 — the audit on Chen et al.'s released logs (Mix set, 600 questions from six sources; their graded token accuracy, %)", "",
          "Told the source: the better of never / always per source, chosen on four folds and applied to the fifth. Differences to it with 95% bootstrap intervals over questions.", "",
          "| reader | never | always | hard gate | soft gate | prompt | human | told the source (in sample) | oracle | hard − told | soft − told | human − told | always − told |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, c in R.items():
        p = c["policies"]
        L.append(f"| {NAMES[r]} | {p['never']['accuracy']} | {p['always']['accuracy']} | {p['hard']['accuracy']} | {p['soft']['accuracy']} | {p['prompt']['accuracy']} | {p['human']['accuracy']} | "
                 f"{p['told the source']['accuracy']} ({c['told_in_sample']}) | {p['oracle']['accuracy']} | {f(p['hard']['vs_told'])} | {f(p['soft']['vs_told'])} | {f(p['human']['vs_told'])} | {f(p['always']['vs_told'])} |")
    L += ["", "Reproduction of their Tables 3 and 11 (Mix, Acc.; never, always, prompt, hard, soft, human): " + "; ".join(f"{NAMES[r]} largest difference {R[r]['reproduction_max_abs_diff']}" for r in printed) + ".",
          f"The hard gate's decisions equal its own majority decision per source on {R['gpt4-o']['hard_agrees_with_its_majority']}% of the questions; "
          f"search rates (hard / soft), by source: " + "; ".join(f"{s} {v['hard_rate']:.0f}% / {v['soft_rate']:.0f}%" for s, v in R["gpt4-o"]["by_source"].items()) + ".",
          "Majority decision per source of the hard / soft gate, difference to told-the-source: " + "; ".join(
              f"{NAMES[r]} {f(c['policies']['hard: majority per source']['vs_told'])} / {f(c['policies']['soft: majority per source']['vs_told'])}" for r, c in R.items()) + ".",
          "200 other fold assignments (hard gate − told on average, share of assignments in which the hard gate is ahead; the same for the soft gate): " + "; ".join(
              f"{NAMES[r]} {c['told_200_assignments']['hard_minus_told_mean']:+.2f}, {c['told_200_assignments']['hard_ahead_share']}%; "
              f"{c['told_200_assignments']['soft_minus_told_mean']:+.2f}, {c['told_200_assignments']['soft_ahead_share']}%" for r, c in R.items()) + ".",
          "\"Human\" is their boundary model trained on human-labelled data, not a person deciding. The accuracy is their token accuracy; their LLM-judge score "
          "(a paid API) is not computed, and under it their Table 4 prints the hard gate above always-search for GPT-4o (54.10 against 52.90).",
          "Told-the-source under five other fold assignments: " + "; ".join(f"{NAMES[r]} {min(c['told_other_seeds'])}–{max(c['told_other_seeds'])}" for r, c in R.items()) + ".", "",
          "| reader | never / always by source (" + ", ".join(rep["sources"]) + ") | helps % | harms % | AUROC floor | needed |", "|---|---|---|---|---|---|"]
    for r, c in R.items():
        L.append(f"| {NAMES[r]} | " + "; ".join(f"{v['never']} / {v['always']}" for v in c["by_source"].values()) + f" | {c['planner']['help']} | {c['planner']['harm']} | {c['planner']['auroc_floor']} | {c['planner']['auroc_needed']} |")
    L += ["", "Expectations: " + "; ".join(f"{k} {'held' if rep[k] else 'NOT held'}" for k in ("K1", "K2", "K3", "K4")) + "."]
    # the demo's second sample log: GPT-4o on the Mix set, an answer correct when its token accuracy is above 0.5 (their
    # threshold), their hard gate's decision as the logged gate; outcomes and decisions only, no question text or image
    if judge:  # PREREG v11b: the validity gate and its own expectations; the demo's sample log is written by the token run only
        rep["judge_valid"] = bool(sum(max(c["never_always_abs_diff"]) <= 5.0 for c in R.values()) >= 4)
        rep["L1"] = bool(lead("gpt4-o", "hard")[0] <= 0)
        rep["L2"] = bool(sum(lead(r, "hard")[0] <= 0 for r in R) >= 4)
        rep["L3"] = bool(not any(lead(r, "hard")[0] >= 1.0 and lead(r, "hard")[1] > 0 for r in R))
        L[0] = "# PREREG v11b — the audit on Chen et al.'s released logs under a judge model's score (Mix set, 600 questions from six sources, %)"
        L += ["", "Judge against their printed scores (never, always; points): " + "; ".join(f"{NAMES[r]} {c['never_always_abs_diff']}" for r, c in R.items())
              + f". Validity gate (within 5.0 points for at least four readers): {'passed' if rep['judge_valid'] else 'NOT passed'}.",
              "Expectations: " + "; ".join(f"{k} {'held' if rep[k] else 'NOT held'}" for k in ("L1", "L2", "L3")) + "."]
        out = ROOT / f"experiments/results/chen_logs_audit__{a.tag}"
        pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    if a.sources:  # a subset: its own result files, and the demo's sample log is left as it is
        L[0] = f"# PREREG v14b — the audit on Chen et al.'s released logs, Mix set restricted to {rep['n']} questions from {len(rep['sources'])} sources ({', '.join(rep['sources'])}); token accuracy, %"
        L = [l for l in L if not l.startswith("Reproduction of their")]
        L += ["", "Expectations (v14b): B1 " + ("held" if all(lead(r, "hard")[0] <= 0 for r in ("gpt4-o", "qwen-vl-max")) else "NOT held")
              + "; B2 " + ("held" if sum(not (lead(r, "hard")[0] >= 1.0 and lead(r, "hard")[1] > 0) for r in R) >= 4 else "NOT held") + "."]
        out = ROOT / f"experiments/results/chen_logs_audit__{a.tag}"
        pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
        print("\n".join(L))
        return
    g4 = d[d.reader == "gpt4-o"].sort_values("row")
    pd.DataFrame(dict(segment=g4.source, group=g4.row, c_ne=(g4.acc_norag > 0.5).astype(int), c_ev=(g4.acc_rag > 0.5).astype(int), decision=g4.hard)
                 ).to_csv(ROOT / "demo/sample_log_published.csv", index=False, lineterminator="\n")
    out = ROOT / "experiments/results/chen_logs_audit"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

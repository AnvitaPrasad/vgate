"""Published retrieve / skip gates through the audit (PREREG v8). Nothing in these gates is fitted by us: each one is a
released prompt or checkpoint, read at its published threshold, and its decisions are applied to our reader's two answers.
Written before the gates were run on our questions (the ReAG-Critic scores of J13, development split, existed).

  python scripts/external_gates.py --split confirm   -> experiments/results/external_gates__confirm.{json,md}
  python scripts/external_gates.py --split dev       (also covers the gates that were run on the development split only)
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import lodo  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

R, DATA, MODEL = ROOT / "experiments/runs", ROOT / "data", "Qwen3-VL-4B-Instruct"
DATASETS = ("evqa", "infoseek", "okvqa")
# name -> (label, run folder, how it is read)
GATES = dict(
    chen=("Search-need prompt of Chen et al. (reader's reply: true / false)", "J16", "prompt"),
    urag=("UniversalRAG router prompt (reader's reply: no / a modality; text query only)", "J16", "prompt"),
    reag_critic=("ReAG-Critic checkpoint, kept passage in the top-5 (after retrieval)", "J13", "critic"),
    reflectiva=("ReflectiVA checkpoint, [Retrieval] token", "J17", "token"),
)
CRITIC_THR = 0.1  # the critic's card keeps a passage when the raw probability of "Yes" exceeds 0.1


def gate_rows(name, d):
    """{qid: (score, decision, extra)} of a gate on one benchmark; {} when it was not run."""
    _, run, kind = GATES[name]
    if kind == "prompt":
        rows = U.read(R / run, d, MODEL, f"gate_{name}")
        return {q: (r["p_retrieve"], r["p_retrieve"] > 0.5, dict(coverage=r["retrieve_raw"] + r["skip_raw"], text=r["text"])) for q, r in rows.items()}
    if kind == "critic":
        per = [U.read(R / run, d, MODEL, f"pv_{k}") for k in range(1, 6)]
        if not all(per):
            return {}
        out = {}
        for q in set.intersection(*(set(p) for p in per)):
            s = max(p[q].get("yes_raw", p[q]["p_yes"]) for p in per)
            out[q] = (s, s > CRITIC_THR, {})
        return out
    rows = U.read(R / run, d, "ReflectiVA", "gate")
    return {q: (r["p_retrieve"], r["p_retrieve"] > 0.5, dict(coverage=r["retrieve_raw"] + r["skip_raw"], text=r.get("text", ""))) for q, r in rows.items()}


def text_decision(name, text):
    """The decision as the reply's words say it (validity check of the first-word probabilities); None if unreadable."""
    t = (text or "").strip().lower().strip("\"'`*. ")
    if name == "chen":
        return True if t.startswith("true") else False if t.startswith("false") else None
    if name == "urag":
        return None if not t else t.split("+")[0].strip() != "no"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="confirm", choices=["dev", "confirm"])
    ap.add_argument("--gates", default=",".join(GATES))
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default=str(ROOT / "experiments/results"))
    a = ap.parse_args()
    runs = lodo.MODELS[MODEL]
    dev = {d: A.load(R / runs["dev"][d], DATA, d, MODEL, "dev") for d in DATASETS}
    te = dev if a.split == "dev" else {d: A.load(R / runs[a.split][d], DATA, d, MODEL, a.split) for d in DATASETS}
    better = {d: bool(dev[d].c_ev.mean() > dev[d].c_ne.mean()) for d in DATASETS}  # the better fixed policy, chosen on development data
    f3 = lambda v: [round(100 * x, 2) for x in v]
    rep = dict(split=a.split, reader=MODEL, better_fixed_reads=better, gates={})
    for name in a.gates.split(","):
        G = {d: gate_rows(name, d) for d in DATASETS}
        if not all(len(set(G[d]) & set(te[d].index)) >= 0.99 * len(te[d]) for d in DATASETS):
            rep["gates"][name] = dict(label=GATES[name][0], status="NOT_RUN on this split",
                                      coverage={d: round(len(set(G[d]) & set(te[d].index)) / len(te[d]), 3) for d in DATASETS})
            continue
        parts, o = {}, dict(label=GATES[name][0], by_benchmark={})
        for d in DATASETS:
            fr = te[d][te[d].index.isin(G[d])]
            sc = np.array([G[d][q][0] for q in fr.index], float); dec = np.array([G[d][q][1] for q in fr.index], bool)
            c0, c1, u = fr.c_ne.to_numpy().astype(float), fr.c_ev.to_numpy().astype(float), fr.u.to_numpy()
            # the gate's own majority decision on this benchmark, taken from the development questions when they were scored
            dq = [q for q in dev[d].index if q in G[d]]
            maj = bool(np.mean([G[d][q][1] for q in dq]) > 0.5) if dq else bool(dec.mean() > 0.5)
            acc = dict(gate=np.where(dec, c1, c0), never=c0, always=c1, fixed=c1 if better[d] else c0, majority=c1 if maj else c0, oracle=np.maximum(c0, c1),
                       random=(1 - dec.mean()) * c0 + dec.mean() * c1)  # expected outcome of random decisions at the gate's read rate (added after the results)
            grp = fr.group.to_numpy()
            diff = lambda x, y, acc=acc, grp=grp: f3(gate.cluster_bootstrap(grp, lambda i: float(acc[x][i].mean() - acc[y][i].mean()), n_boot=a.n_boot, alpha=0.05))
            b = dict(n=len(fr), read=round(100 * float(dec.mean()), 1), **{k: round(100 * float(v.mean()), 2) for k, v in acc.items()},
                     gate_minus_fixed=diff("gate", "fixed"), gate_minus_random=diff("gate", "random"), majority_reads=maj, same_as_majority=round(100 * float((dec == maj).mean()), 1),
                     auroc_help_vs_harm=round(gate.auroc(sc[u != 0], u[u != 0] > 0), 3), auroc_closed_book_wrong=round(gate.auroc(sc, c0 == 0), 3))
            ex = [G[d][q][2] for q in fr.index]
            if ex and "coverage" in ex[0]:
                td = [text_decision(name, e["text"]) for e in ex]
                b.update(word_mass=round(float(np.mean([e["coverage"] for e in ex])), 3), reply_unreadable=round(100 * float(np.mean([t is None for t in td])), 1),
                         reply_agrees=round(100 * float(np.mean([t == x for t, x in zip(td, dec) if t is not None] or [np.nan])), 1))
            o["by_benchmark"][d] = b; parts[d] = (acc, grp)
        offs = np.cumsum([0] + [len(parts[d][1]) for d in DATASETS])
        grp = np.concatenate([parts[d][1] for d in DATASETS]); strata = np.concatenate([np.full(len(parts[d][1]), j) for j, d in enumerate(DATASETS)])

        def macro(x, y=None):
            def stat(idx):
                v = []
                for j, d in enumerate(DATASETS):
                    i = idx[(idx >= offs[j]) & (idx < offs[j + 1])] - offs[j]
                    v.append(parts[d][0][x][i].mean() - (parts[d][0][y][i].mean() if y else 0.0))
                return float(np.mean(v))
            return f3(gate.cluster_bootstrap(grp, stat, n_boot=a.n_boot, alpha=0.05, strata=strata))
        o["mixed"] = dict(**{k: macro(k)[0] for k in ("gate", "never", "always", "fixed", "majority", "oracle", "random")},
                          gate_minus_random_at_its_read_rates=macro("gate", "random"),
                          gate_minus_told_the_benchmark=macro("gate", "fixed"), gate_minus_always=macro("gate", "always"),
                          gate_minus_its_benchmark_majority=macro("gate", "majority"),
                          same_as_majority=round(float(np.mean([o["by_benchmark"][d]["same_as_majority"] for d in DATASETS])), 1))
        o["expectations"] = dict(X1=bool(all(o["by_benchmark"][d]["gate_minus_fixed"][0] < 1.0 for d in DATASETS)),
                                 X2=bool(o["mixed"]["gate_minus_told_the_benchmark"][2] < 1.0),
                                 X3=bool(o["mixed"]["gate_minus_its_benchmark_majority"][0] < 1.0))  # one-sided, see the amendment to v8
        rep["gates"][name] = o
    f = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
    L = [f"# Published gates through the audit (PREREG v8) — {a.split} split, reader {MODEL}, 95% image-cluster intervals", "",
         "Each gate is used as released (prompt or checkpoint, published threshold); its decisions are applied to the reader's closed-book and top-5 answers.", ""]
    for name, o in rep["gates"].items():
        L += [f"## {o['label']}", ""]
        if "status" in o:
            L += [f"{o['status']} (share of questions scored: {o['coverage']}).", ""]
            continue
        L += ["| | " + " | ".join(DATASETS) + " |", "|---|" + "---|" * len(DATASETS)]
        for k, nm in (("read", "reads (%)"), ("gate", "accuracy following the gate"), ("never", "never"), ("always", "always"), ("fixed", "better fixed policy"),
                      ("oracle", "oracle gate"), ("same_as_majority", "decisions equal to its majority on the benchmark (%)"),
                      ("auroc_help_vs_harm", "AUROC of its score, help vs harm"), ("auroc_closed_book_wrong", "AUROC of its score, closed-book answer wrong"),
                      ("word_mass", "probability mass on the expected words"), ("reply_unreadable", "reply not readable (%)"), ("reply_agrees", "reply agrees with the probability rule (%)")):
            if k in o["by_benchmark"][DATASETS[0]]:
                L.append(f"| {nm} | " + " | ".join(str(o["by_benchmark"][d][k]) for d in DATASETS) + " |")
        L += ["| gate − better fixed policy | " + " | ".join(f(o["by_benchmark"][d]["gate_minus_fixed"]) for d in DATASETS) + " |",
              "| gate − random decisions at its read rate (exploratory) | " + " | ".join(f(o["by_benchmark"][d]["gate_minus_random"]) for d in DATASETS) + " |", "",
              f"Mixed traffic (mean of the three): gate {o['mixed']['gate']}, never {o['mixed']['never']}, always {o['mixed']['always']}, told the benchmark {o['mixed']['fixed']}, "
              f"oracle {o['mixed']['oracle']}. Gate − told the benchmark {f(o['mixed']['gate_minus_told_the_benchmark'])}; gate − always {f(o['mixed']['gate_minus_always'])}; "
              f"gate − its own majority decision per benchmark {f(o['mixed']['gate_minus_its_benchmark_majority'])} ({o['mixed']['same_as_majority']}% of decisions equal that majority); "
              f"gate − random decisions at its read rate on each benchmark {f(o['mixed']['gate_minus_random_at_its_read_rates'])} (exploratory, added after the results).", "",
              "Expectations: " + ", ".join(f"{k} {'held' if v else 'NOT held'}" for k, v in o["expectations"].items()) + ".", ""]
    out = pathlib.Path(a.out) / f"external_gates__{a.split}"
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=float)); out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

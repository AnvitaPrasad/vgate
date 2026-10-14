"""Exploratory, DEVELOPMENT data only: the after-reading rule of PREREG v7c on the three open-ended benchmarks (answer with
the evidence reply iff its first-token probability exceeds that of the closed-book reply). Gain over the better fixed policy
of each benchmark, 80% image-cluster intervals, four readers.

  python scripts/select_open_ended_dev.py   -> experiments/results/select_open_ended__dev.{json,md}
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402
import lodo  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

R, DATA = ROOT / "experiments/runs", ROOT / "data"
rep = {}
for model, runs in lodo.MODELS.items():
    rep[model] = {}
    for d, run in runs["dev"].items():
        fr = A.load(R / run, DATA, d, model, "dev")
        ne, ev = U.read(R / run, d, model, "ne"), U.read(R / run, d, model, "ev")
        ok = [q for q in fr.index if q in ne and q in ev and ne[q].get("tok_logprobs") and ev[q].get("tok_logprobs")]
        fr = fr.loc[ok]
        c0, c1, g = fr.c_ne.to_numpy().astype(float), fr.c_ev.to_numpy().astype(float), fr.group.to_numpy()
        best = c1 if c1.mean() > c0.mean() else c0
        use = np.array([ev[q]["tok_logprobs"][0] > ne[q]["tok_logprobs"][0] for q in ok])
        a_ = np.where(use, c1, c0)
        v = gate.cluster_bootstrap(g, lambda i: float(a_[i].mean() - best[i].mean()), n_boot=2000, alpha=0.2)
        rep[model][d] = dict(n=len(ok), never=round(100 * c0.mean(), 1), always=round(100 * c1.mean(), 1), rule=round(100 * a_.mean(), 1),
                             uses_evidence_answer=round(100 * float(use.mean()), 1), rule_minus_better_fixed=[round(100 * x, 1) for x in v])
f = lambda v: f"{v[0]:+.1f} [{v[1]:+.1f}, {v[2]:+.1f}]"
L = ["# The after-reading rule on the open-ended benchmarks (exploratory; development split; 80% intervals)", "",
     "Gain over the better fixed policy of each benchmark.", "", "| reader | E-VQA | InfoSeek | OK-VQA |", "|---|---|---|---|"]
L += [f"| {m} | " + " | ".join(f(rep[m][d]["rule_minus_better_fixed"]) for d in ("evqa", "infoseek", "okvqa")) + " |" for m in rep]
out = ROOT / "experiments/results/select_open_ended__dev"
out.with_suffix(".json").write_text(json.dumps(rep, indent=1)); out.with_suffix(".md").write_text("\n".join(L) + "\n")
print("\n".join(L))

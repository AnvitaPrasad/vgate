"""CVQA: choosing the answer after reading (PREREG v7c). No fitted parameter: answer with the evidence reply iff it names
an option and its first-token probability exceeds that of the closed-book reply; otherwise answer closed-book.

  python scripts/cvqa_select.py [--split confirm|dev]   -> experiments/results/cvqa_select__<split>.{json,md}
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import cvqa_audit as CA  # noqa: E402
from vgate import gate, uncertainty as U  # noqa: E402

R = ROOT / "experiments/runs"
DATA = R / "J14/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="confirm", choices=["dev", "confirm"])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--model", default=CA.MODEL, help="reader whose outputs are analysed (PREREG v7d: a second reader)")
    ap.add_argument("--run_dev", default=str(R / "J14")); ap.add_argument("--run_confirm", default=str(R / "J15"))
    ap.add_argument("--fallback", action="store_true", help="registered with v7c: the same comparisons on the fallback labels of v7 "
                    "(an evidence reply that names no option counts as the closed-book answer, for always-retrieve and the gates too)")
    a = ap.parse_args()
    tag = ("__fallback" if a.fallback else "") + ("" if a.model == CA.MODEL else f"__{a.model}")
    CA.MODEL = a.model
    run = pathlib.Path(a.run_dev if a.split == "dev" else a.run_confirm)
    alpha = 0.2 if a.split == "dev" else 0.05
    fr, H, seg = CA.load(run, DATA, a.split, a.fallback)
    ne, ev = U.read(run, "cvqa", CA.MODEL, "ne"), U.read(run, "cvqa", CA.MODEL, "ev")
    p0 = np.array([ne[q]["tok_logprobs"][0] for q in fr.index]); p1 = np.array([ev[q]["tok_logprobs"][0] for q in fr.index])
    use_ev = (p1 > p0) & ~fr.no_letter_ev.to_numpy()
    c0, c1, grp = fr.c_ne.to_numpy().astype(float), fr.c_ev.to_numpy().astype(float), fr.group.to_numpy()
    acc = dict(never=c0, always=c1, select=np.where(use_ev, c1, c0), oracle=np.maximum(c0, c1),
               prob_only=np.where(p1 > p0, c1, c0))  # the probability comparison without the names-an-option condition
    if a.split == "confirm":  # the pre-read gate of v7, fitted on the development split
        tr, Htr, seg_tr = CA.load(a.run_dev, DATA, "dev", a.fallback)
        D, _ = CA.decide(tr, Htr, fr, H, seg_tr, seg)
        acc["vgate_full"] = np.where(D["benefit_full"], c1, c0); acc["vgate_post"] = np.where(D["benefit_post"], c1, c0)
    diff = lambda x, y: [round(100 * v, 2) for v in gate.cluster_bootstrap(grp, lambda i: float(acc[x][i].mean() - acc[y][i].mean()), n_boot=a.n_boot, alpha=alpha)]
    tok0, tok1 = fr.tok_ne.to_numpy(), fr.tok_ev.to_numpy()
    rep = dict(split=a.split, n=len(fr), interval=f"{round(100 * (1 - alpha))}%", acc={k: round(100 * float(v.mean()), 2) for k, v in acc.items()},
               uses_evidence_answer=round(100 * float(use_ev.mean()), 1), select_minus_never=diff("select", "never"), select_minus_always=diff("select", "always"),
               tokens_relative_to_always=round(float((tok0 + tok1).mean() / tok1.mean()), 2),
               help_kept=round(100 * float(use_ev[(c1 > c0)].mean()), 1), harm_avoided=round(100 * float((~use_ev)[(c1 < c0)].mean()), 1),
               fallback_labels=bool(a.fallback), prob_only_minus_never=diff("prob_only", "never"),
               no_option_but_more_confident=round(100 * float(((p1 > p0) & fr.no_letter_ev.to_numpy()).mean()), 1))
    if "vgate_full" in acc:
        rep["select_minus_vgate_full"] = diff("select", "vgate_full"); rep["select_minus_vgate_post"] = diff("select", "vgate_post")
        rep["expectations"] = dict(S1=bool(rep["select_minus_never"][0] >= 1.0 and rep["select_minus_never"][1] > 0), S2=bool(rep["select_minus_vgate_full"][1] > 0))
    f = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
    L = [f"# CVQA: choosing the answer after reading (PREREG v7c) — {a.split} split, n = {rep['n']}, {rep['interval']} image-cluster intervals"
         + (" — FALLBACK LABELS (a reply that names no option counts as the closed-book answer for every policy)" if a.fallback else ""), "",
         f"Accuracy: never-retrieve {rep['acc']['never']}, always-retrieve {rep['acc']['always']}, the rule {rep['acc']['select']}"
         + (f", VGate-full {rep['acc']['vgate_full']}" if "vgate_full" in acc else "") + f", best of the two answers {rep['acc']['oracle']}.",
         f"The rule uses the evidence answer on {rep['uses_evidence_answer']}% of questions; it keeps {rep['help_kept']}% of the cases where retrieval helps and avoids "
         f"{rep['harm_avoided']}% of the cases where it harms. Tokens relative to always-retrieve: {rep['tokens_relative_to_always']}.", "",
         f"- rule − never-retrieve (the better fixed policy): {f(rep['select_minus_never'])}", f"- rule − always-retrieve: {f(rep['select_minus_always'])}",
         f"- the probability comparison alone, without the names-an-option condition − never-retrieve: {f(rep['prob_only_minus_never'])} "
         f"(the evidence reply names no option and is still the more confident one on {rep['no_option_but_more_confident']}% of questions)"]
    if "vgate_full" in acc:
        L += [f"- rule − VGate-full: {f(rep['select_minus_vgate_full'])}", f"- rule − VGate-post (not a registered comparison): {f(rep['select_minus_vgate_post'])}",
              "", "## Expectations (PREREG v7c)", ""] + [f"- {k}: {'held' if v else 'NOT held'}" for k, v in rep["expectations"].items()]
    out = ROOT / "experiments/results" / f"cvqa_select__{a.split}{tag}"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

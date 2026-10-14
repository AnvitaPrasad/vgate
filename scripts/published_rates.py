"""The planner on help and harm counts that another paper prints for a much larger reader (no model pass of ours).

Xu, Moor and Leskovec, "Reverse Image Retrieval Cues Parametric Memory in Multimodal LLMs" (arXiv:2405.18740v1): GPT-4o on
1,650 InfoSeek questions (150 per category, Sec. 4.1), closed-book against always given a screenshot of reverse-image-search
results. Table 3 (App. A.1.2) prints the helped and the harmed questions per category; Table 1 the two accuracies; Table 4
(App. A.1.6) three GPT-4o-based agents that decide whether to use the search, with the share of the helped (a) and of the
harmed (b) questions each one gets wrong, and their accuracy. Numbers below are copied from those tables.

  python scripts/published_rates.py   -> experiments/results/published_rates.{json,md}
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vgate import gate  # noqa: E402

N = 1650
TABLE3 = dict(facility=(41, 5), building=(43, 7), location=(27, 3), plant=(17, 5), others=(19, 9), sport=(14, 7), vehicle=(17, 12),
              organization_and_company=(8, 3), objects=(13, 10), animal=(15, 15), food=(7, 18))  # category: (helping, hurting)
TABLE1 = dict(closed_book=39.21, always=46.91)  # GPT-4o, GPT-as-judge accuracy
TABLE4 = {"Decide + Consistency Check": (57.92, 28.72, 43.21), "Decide only": (17.19, 67.02, 46.48), "Consistency Check only": (53.85, 39.36, 43.15)}  # a %, b %, accuracy %


def main():
    p, q = sum(v[0] for v in TABLE3.values()), sum(v[1] for v in TABLE3.values())
    h, m = p / N, q / N
    assert abs(100 * (h - m) - (TABLE1["always"] - TABLE1["closed_book"])) < 0.01  # Table 3's net gain is Table 1's accuracy difference
    rep = dict(source="arXiv:2405.18740v1, Tables 1, 3, 4", reader="GPT-4o", benchmark="InfoSeek (1,650 questions, 150 per category)",
               evidence="a screenshot of reverse-image-search results", n=N, helped=p, harmed=q, help=round(100 * h, 2), harm=round(100 * m, 2),
               oracle_lead=round(100 * min(h, m), 2), auroc_floor=round(gate.auroc_needed(h, m, shape="any"), 3), auroc_needed=round(gate.auroc_needed(h, m), 3), agents={})
    for name, (a, b, acc) in TABLE4.items():
        lead = 100 * (m * (1 - b / 100) - h * a / 100)  # the gain identity against always-retrieve: harmed questions skipped minus helped questions skipped
        rep["agents"][name] = dict(reads_of_helped=round(100 - a, 2), reads_of_harmed=round(b, 2), auroc_of_the_decision=round((1 - a / 100 + 1 - b / 100) / 2, 3),
                                   lead_from_identity=round(lead, 2), accuracy_from_identity=round(TABLE1["always"] + lead, 2), accuracy_printed=acc)
    rep["identity_max_abs_error"] = round(max(abs(v["accuracy_from_identity"] - v["accuracy_printed"]) for v in rep["agents"].values()), 2)
    rep["agents_below_floor"] = sum(v["auroc_of_the_decision"] < rep["auroc_floor"] for v in rep["agents"].values())
    L = ["# The planner on published counts for a larger reader (GPT-4o on InfoSeek; arXiv:2405.18740v1, Tables 1, 3, 4)", "",
         f"{N} questions; evidence: {rep['evidence']}. Helped {p} ({rep['help']}%), harmed {q} ({rep['harm']}%) (sums of Table 3's eleven categories; "
         f"their difference is Table 1's {TABLE1['closed_book']} -> {TABLE1['always']}). An oracle gate would lead always-retrieve by {rep['oracle_lead']} points; "
         f"a one-point lead needs a help-vs-harm AUROC of at least {rep['auroc_floor']} on a concave curve and {rep['auroc_needed']} at the binormal estimate.", "",
         "| agent (Table 4) | reads % of helped | reads % of harmed | AUROC of the one decision | lead over always-retrieve from the identity | accuracy from the identity | printed |", "|---|---|---|---|---|---|---|"]
    L += [f"| {k} | {v['reads_of_helped']} | {v['reads_of_harmed']} | {v['auroc_of_the_decision']} | {v['lead_from_identity']:+.2f} | {v['accuracy_from_identity']} | {v['accuracy_printed']} |"
          for k, v in rep["agents"].items()]
    L += ["", f"The identity reproduces the three printed accuracies within {rep['identity_max_abs_error']} points; {rep['agents_below_floor']} of 3 decisions are below the floor.",
          "Not comparable with our setting in every respect: the evidence is an image of a results page from a live search, not text passages; a GPT-4-Turbo judge; one run."]
    out = ROOT / "experiments/results/published_rates"
    pathlib.Path(str(out) + ".json").write_text(json.dumps(rep, indent=1)); pathlib.Path(str(out) + ".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

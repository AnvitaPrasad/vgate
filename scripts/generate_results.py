"""Paper artifacts from a confirmatory JSON (scripts/confirm.py): Fig. 2 (frontiers) and Table 1 (LaTeX).

  python scripts/generate_results.py experiments/runs/J4/confirm__Qwen3-VL-4B-Instruct.json [--out paper]
Every number printed in the paper's table comes from this JSON; nothing is typed by hand.
Palette: dataviz reference categorical slots 1-4 in fixed order (validated on white: all hard checks pass;
aqua/yellow < 3:1 contrast -> relieved by legend + direct end-labels + distinct dashes + Table 1).
"""
import argparse
import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GRID = np.linspace(0, 1, 101)
SERIES = [  # (policy key, label, color, dash) -- fixed categorical order, never cycled
    ("benefit_post", "VGate-post (benefit)", "#2a78d6", "-"),
    ("error_post", "Error gate (post-search)", "#eb6834", "-"),
    ("probe_benefit", "VGate-pre (probe)", "#1baf7a", (0, (4, 1.5))),
    ("signal_ent_mean", "Entropy threshold", "#eda100", (0, (1.2, 1.2))),
]
ORACLE = ("oracle", "Oracle gate", "#52514e", (0, (3, 2)))
NAMES = {"evqa": "E-VQA", "infoseek": "InfoSeek", "okvqa": "OK-VQA"}
SHORT = {"evqa": "E-VQA", "infoseek": "InfoS.", "okvqa": "OK-VQA"}
ROWS = [("random", "Random"), ("signal_ent_mean", "Entropy threshold"), ("probe_error", "Error probe (hidden state)"),
        ("probe4", "Four-outcome probe"), ("probe_benefit", "\\textbf{VGate-pre} (benefit probe)"),
        ("error_post", "Error gate (unc.\\,+\\,retriever)"), ("signal_qpp_nqc", "NQC threshold (retriever)"),
        ("factorised_post", "Factorised P(wrong)$\\cdot$P(hit)"), ("benefit_post", "\\textbf{VGate-post} (benefit)"),
        ("vgate_full", "\\textbf{VGate-full} (+\\,hidden state)"), ("oracle", "Oracle gate")]
STAGE_BREAKS = {"random": "\\multicolumn{7}{l}{\\emph{Pre-search (decides search and read)}}",
                "error_post": "\\multicolumn{7}{l}{\\emph{Post-search (always searches; decides read)}}",
                "oracle": None}


def fig(rep, datasets, path):
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#8a8986", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "font.family": "sans-serif"})
    f, axes = plt.subplots(1, len(datasets), figsize=(3.4, 1.3), sharey=False)
    for ax, d in zip(np.atleast_1d(axes), datasets):
        P = rep[d]["policies"]
        for key, label, color, dash in SERIES + [ORACLE]:
            if key in P:
                acc = np.array(P[key]["curve_acc"]) * 100
                ax.plot(GRID, acc, color=color, lw=1.6 if key == "benefit_post" else 1.1, ls=dash, label=label,
                        zorder=3 if key == "benefit_post" else 2)
        q = rep[d]["quadrants"]
        ax.scatter([0], [q["acc_never"][0] * 100], marker="s", s=10, color="#52514e", zorder=4)
        ax.scatter([1], [q["acc_always"][0] * 100], marker="^", s=12, color="#52514e", zorder=4)
        up = q["acc_always"][0] > q["acc_never"][0]  # label placement: keep clear of the curves in both regimes
        ax.annotate("never", (0, q["acc_never"][0] * 100), xytext=(4, -3 if up else -8), textcoords="offset points", fontsize=5, color="#52514e")
        ax.annotate("always", (1, q["acc_always"][0] * 100), xytext=(-17, -8 if up else 4), textcoords="offset points", fontsize=5, color="#52514e")
        ax.set_title(NAMES.get(d, d), fontsize=6.5, color="#0b0b0b", weight="bold", loc="left", pad=2)
        ax.grid(color="#e8e7e3", lw=0.4); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("accuracy (%)", fontsize=6, color="#52514e")
    f.supxlabel("fraction of queries that read evidence", fontsize=6, color="#52514e", y=0.02)
    h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
    f.legend(h, l, loc="upper center", ncol=3, fontsize=5.5, frameon=False, bbox_to_anchor=(0.5, 1.2))
    f.tight_layout(pad=0.3)
    f.savefig(path, bbox_inches="tight")
    f.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")


def table(rep, datasets, path):
    """nAUC on every dataset; Read (share of reads needed to come within 0.5 points of always-retrieve) only for the
    datasets where always-retrieve beats never-retrieve (it is 0% by construction elsewhere)."""
    read_ds = [d for d in datasets if rep[d]["quadrants"]["acc_always"][0] > rep[d]["quadrants"]["acc_never"][0]]
    ncol, eol = len(datasets) + len(read_ds), " \\\\"
    lines = ["\\begin{tabular}{@{}l" + "r" * ncol + "@{}}", "\\toprule",
             f" & \\multicolumn{{{len(datasets)}}}{{c}}{{nAUC}} & \\multicolumn{{{len(read_ds)}}}{{c}}{{Read}}" + eol,
             "Policy & " + " & ".join(SHORT.get(d, d) for d in datasets + read_ds) + eol]
    for key, label in ROWS:
        if key in STAGE_BREAKS:
            lines.append("\\midrule")
            if STAGE_BREAKS[key]:
                lines.append(STAGE_BREAKS[key].replace("{7}", "{%d}" % (ncol + 1)) + eol)
        cells = []
        for d in datasets:
            v = rep[d]["policies"].get(key)
            cells.append("--" if v is None else f"{100 * v['nauc']:.1f}")
        for d in read_ds:
            v = rep[d]["policies"].get(key)
            sr = v["read_rate_at_always_acc"] if v else float("nan")
            cells.append("--" if sr != sr else f"{100 * sr:.0f}\\%")
        lines.append(f"{label} & " + " & ".join(cells) + eol)
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("confirm_json")
    ap.add_argument("--extra", default="", help="confirm JSON of further datasets (e.g. the OK-VQA run J6)")
    ap.add_argument("--out", default="paper")
    a = ap.parse_args()
    rep = {}
    for cj in [a.confirm_json] + ([a.extra] if a.extra else []):
        r = json.load(open(cj))
        ext = pathlib.Path(cj.replace("confirm__", "extend__"))  # PREREG v2b gates (scripts/extend.py)
        if ext.exists():
            for d, v in json.load(open(ext)).items():
                if d in r and "policies" in v:
                    r[d]["policies"].update(v["policies"])
        rep.update({k: v for k, v in r.items() if k not in rep})
    datasets = [d for d in ("evqa", "infoseek", "okvqa") if d in rep]
    out = pathlib.Path(a.out)
    (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(parents=True, exist_ok=True)
    fig(rep, datasets, out / "figures" / "frontier.pdf")
    table(rep, datasets, out / "tables" / "main.tex")
    p = rep["primary"]
    (out / "tables" / "primary.tex").write_text(
        f"\\newcommand{{\\primarypt}}{{{p['point_pt']:+.2f}}}\n\\newcommand{{\\primarylo}}{{{p['ci95_pt'][0]:+.2f}}}\n"
        f"\\newcommand{{\\primaryhi}}{{{p['ci95_pt'][1]:+.2f}}}\n")
    print("wrote", out / "figures/frontier.pdf", out / "tables/main.tex", out / "tables/primary.tex")


if __name__ == "__main__":
    main()

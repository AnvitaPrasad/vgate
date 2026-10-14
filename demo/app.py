"""VGate demo (cached mode): every retrieval decision, its predicted benefit, its cost, and the harm it avoids.

  python demo/app.py [--run experiments/runs/J4 --split confirm] [--share]

Reads the run's gate_scores__*.csv (written by scripts/analyze.py), the processed questions and the run's JSONL
outputs; nothing is recomputed on the fly, so the numbers shown are exactly the ones in the paper's tables. The one
exception is the "Learn from feedback" tab, which re-runs the simulation of scripts/online_gate.py on the logged runs.
"""
import argparse
import tempfile
import json
import os
import pathlib
import re
import sys

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vgate import gate, uncertainty as U  # noqa: E402

POLICIES = {"vgate_full": "VGate-full (benefit; uncertainty + retriever + hidden state)",
            "benefit_post": "VGate-post (benefit; uncertainty + retriever)", "probe_benefit": "VGate-pre (hidden-state probe)",
            "factorised_post": "Factorised P(wrong)·P(hit)", "error_post": "Error gate (VGate-post's features)",
            "signal_ent_mean": "Entropy threshold", "oracle": "Oracle gate"}
READ_GATES = ["vgate_full", "benefit_post"]  # stage 2 (read?): VGate-full when scripts/extend.py has been run
PRE_GATE = "probe_benefit"  # stage 1 (search?): the paper's VGate-pre
GATE_NAME = {"vgate_full": "VGate-full", "benefit_post": "VGate-post"}
SAMPLE_LOG = ROOT / "demo/sample_log.csv"  # scripts/make_sample_log.py: outcomes and signals only, no benchmark content
# scripts/chen_logs_audit.py: GPT-4o on the Mix set of Chen et al. (arXiv:2502.18023, their released answers; Apache-2.0), with
# the search decisions of their fine-tuned gate as the logged gate; an answer counts as correct above 0.5 token accuracy
PUBLISHED_LOG = ROOT / "demo/sample_log_published.csv"
ABSTAIN = re.compile(r"\bnot (specified|available|answerable|determinable|mentioned|provided|given|stated|known|clear)"
                     r"|\bnone\b|\bunknown\b|cannot|can't", re.I)


def failure_mode(ev_text, evidence):
    """Audit-derived harm taxonomy: abstention, distractor-entity capture, or other wrong answer."""
    ev_text = ev_text if isinstance(ev_text, str) else ""
    if not ev_text.strip() or ABSTAIN.search(ev_text):
        return "evidence-induced abstention"
    t = ev_text.lower().strip(" .")
    if t and any(t in e["title"].lower() or e["title"].lower() in t for e in evidence if len(e["title"]) > 3):
        return "distractor-entity capture"
    return "other wrong answer"


FOOTER = ("Passage text is excerpted (truncated) from Wikipedia and is available under CC BY-SA "
          "(https://creativecommons.org/licenses/by-sa/3.0/); the source column links to each article. Questions and "
          "reference answers are from Encyclopedic-VQA, InfoSeek and OK-VQA. Image credits appear under the question. "
          "The benchmark tabs replay logged model runs; only the Live tab, when present, runs the model.")


def wiki_url(e):
    """Source article of an evidence item (InfoSeek ids are URLs; the others are addressed by title)."""
    i = str(e.get("id", ""))
    return i if i.startswith("http") else "https://en.wikipedia.org/wiki/" + str(e["title"]).strip().replace(" ", "_")


SIGNALS = ["ent_mean", "lp_min", "sample_agree", "ent_agree", "p_true", "para_agree", "qpp_top1_z", "qpp_margin_z"]


class Store:
    def __init__(self, run, model, split, data=ROOT / "data", public=False):
        self.model, self.split, self.public = model, split, public
        self.ds = {}
        ap_ = data / "gld_attribution_evqa.csv"  # credits of the CC BY images (scripts/fetch_gld_attribution.py)
        self.credit = pd.read_csv(ap_, index_col="id").to_dict("index") if ap_.exists() else {}
        for run in map(pathlib.Path, str(run).split(",")):  # several run folders, e.g. J4 (E-VQA, InfoSeek) and J6 (OK-VQA)
            for ds in ["evqa", "infoseek", "okvqa"]:
                p = run / f"gate_scores__{ds}__{model}__{split}.csv"
                if not p.exists() or ds in self.ds:
                    continue
                g = pd.read_csv(p, index_col="qid")
                g.index = g.index.astype(str)
                recs = {r["qid"]: r for r in map(json.loads, open(data / "processed" / f"{ds}.jsonl")) if r["qid"] in g.index}
                feats = U.build(run, ds, model, list(recs.values()))
                ap = run / f"analysis__{model}__{split}.json"
                pre = json.loads(ap.read_text())[ds]["feature_groups_pre"] if ap.exists() else ["free"]  # confirm: pruned
                self.ds[ds] = dict(g=g, recs=recs, feats=feats, pre=pre)
        self.gate = next(k for k in READ_GATES if all(f"score_{k}" in S["g"] for S in self.ds.values())) if self.ds else READ_GATES[0]
        img_roots = [ROOT / "data/raw/evqa_images", ROOT / "data/raw/infoseek_images", ROOT / "data/raw/okvqa_images"]
        self.img = {f: os.path.join(dp, f) for r in img_roots for dp, _, fs in os.walk(r) for f in fs}

    def image_credit(self, rec):
        """Markdown credit line for images that carry one (Google Landmarks v2 train, CC BY), else ''."""
        c = self.credit.get(pathlib.Path(rec["image_file"]).stem)
        if not c:
            return ""
        lic = re.sub(r"\(https?://[^)]*\)", "", str(c["license"])).strip()
        title = re.sub(r"\.\w{3,4}$", "", re.sub(r"^File:", "", str(c["title"])))
        title = title[:36] + ("…" if len(title) > 36 else "")
        return f"Image: [{title}]({c['url']}) by {str(c['author']).strip()}, {lic}"

    def can_show(self, rec):
        """Public mode shows only images whose terms allow redistribution with credit (see data/README.md)."""
        return (not self.public) or bool(self.image_credit(rec))

    def qids(self, ds):
        S = self.ds[ds]
        ok = [q for q in sorted(S["g"].index) if self.can_show(S["recs"][q])]
        return ok or sorted(S["g"].index)  # datasets without creditable images stay browsable, without the image

    def threshold(self, ds, policy, budget, auto=False):
        """Score threshold that retrieves the top `budget` fraction for this policy; with `auto`, the budget-free
        rule of a benefit gate: read iff the predicted benefit P(help) - P(harm) is positive."""
        if auto:
            return 0.0
        s = self.ds[ds]["g"][f"score_{policy}"].to_numpy()
        return np.quantile(s, 1 - budget) if budget < 1 else -np.inf

    def frontier_fig(self, ds, budget, unit="tokens"):
        g = self.ds[ds]["g"]
        fig, ax = plt.subplots(figsize=(6, 4))
        always = g.tok_ev.mean()
        for pol, label in POLICIES.items():
            if f"score_{pol}" not in g:
                continue
            pre = self.ds[ds]["pre"]
            groups = {"vgate_full": pre + ["qpp"], "benefit_post": pre + ["qpp"], "error_post": pre + ["qpp"], "factorised_post": pre + ["qpp"],
                      "benefit_pre": pre}.get(pol, ["free"])  # probes use the free NE pass
            base = U.base_cost(g, groups).to_numpy()
            acc, cost = gate.curve(g[f"score_{pol}"].to_numpy(), g.c_ne.to_numpy(), g.c_ev.to_numpy(), base,
                                   g.tok_ev.to_numpy())
            x = gate.GRID if unit == "reads" else cost / always
            ax.plot(x, acc * 100, label=label, lw=2 if pol == self.gate else 1.2,
                    ls="--" if pol == "oracle" else "-")
            if pol == self.gate:
                i = int(round(budget * 100))
                ax.scatter([x[i]], [acc[i] * 100], s=60, zorder=5, color="black")
                k = int(round(100 * float((g[f"score_{pol}"] > 0).mean())))  # budget-free operating point
                ax.scatter([x[k]], [acc[k] * 100], s=110, zorder=6, marker="*", color="black", label="read iff predicted benefit > 0")
        ax.scatter([0.0 if unit == "reads" else g.tok_ne.mean() / always], [g.c_ne.mean() * 100], marker="s",
                   color="gray", label="Never retrieve")
        ax.scatter([1.0], [g.c_ev.mean() * 100], marker="^", color="gray", label="Always retrieve")
        ax.set_xlabel("fraction of queries that read evidence" if unit == "reads" else "total VLM tokens / always-retrieve tokens")
        ax.set_ylabel("accuracy (%)")
        ax.grid(alpha=.3); ax.legend(fontsize=7, loc="lower right"); fig.tight_layout()
        return fig


def live_views(res):
    """Markdown + tables for one result of vgate.live.LivePipeline.answer (see that module for the fields)."""
    stage1 = (f"**Stage 1 — search?** VGate-pre: predicted benefit {res['u_pre']:+.3f} → "
              f"{'SEARCH' if res['search'] else 'SKIP SEARCH'}")
    stage2 = ""
    if "u_full" in res:
        stage2 = (f"\n\n**Stage 2 — read?** VGate-full: predicted benefit {res['u_full']:+.3f} → "
                  + ("READ" if res["read"] else "SKIP READ" if res["search"] else "(not reached: stage 1 skipped the search)")
                  + f"\n\nsearch query: *{res['caption']}* + the question")
    head = "🔎 READ EVIDENCE" if res["read"] else "✅ ANSWER DIRECTLY"
    dec = f"### Decision: {head}\n{stage1}{stage2}\n\nfinal answer: **{res['final']}**"
    rows = [{"policy": "Never retrieve", "answer": res["a0"], "tokens": res["tok_closed_book"]}]
    if "aE" in res:
        rows.append({"policy": "Always retrieve", "answer": res["aE"],
                     "tokens": res["tok_closed_book"] + res.get("tok_caption", 0) + res["tok_evidence"]})
    rows.append({"policy": "VGate", "answer": res["final"], "tokens": res["tok_vgate"]})
    ev = pd.DataFrame([{"rank": i + 1, "title": e["title"], "score": round(e["score"], 3), "source": wiki_url(e),
                        "text": e.get("text", "")[:300]} for i, e in enumerate(res.get("evidence", []))])
    secs = " · ".join(f"{k} {v:.2f}s" for k, v in res.get("seconds", {}).items())
    return dec, pd.DataFrame(rows), ev, f"<small>wall time: {secs}</small>"


def feedback_sim(model, split):
    """Simulation behind the 'Learn from feedback' tab (scripts/online_gate.py, PREREG v3b): the gate is learned on a
    stream of the logged held-out queries from feedback on the shown answer. None if the frozen gates or runs are missing."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import online_gate as OG
        from vgate.live import GATES_DIR, Gates
        if model not in OG.RUNS_BY_MODEL:
            raise KeyError(f"the simulation is set up for {sorted(OG.RUNS_BY_MODEL)}")
        X, c0, c1, _, uf = OG.stream(split, Gates(GATES_DIR / f"{model}.npz"), model)
    except (ImportError, OSError, KeyError) as e:
        print(f"Learn-from-feedback tab disabled: {e}")
        return None

    def sim(eps, both, seed):
        got, act, order = OG.run(X, c0, c1, int(seed), both=bool(both), eps=float(eps))
        third = len(got) // 3
        fig, ax = plt.subplots(figsize=(6, 4))
        for y, label, kw in ((got, "gate learned on this stream", dict(lw=2, color="black")),
                             (np.where(uf > 0, c1, c0)[order], "gate fitted on development logs (both answers judged)", dict(lw=1.2)),
                             (c1[order], "always retrieve", dict(lw=1, ls="--")), (c0[order], "never retrieve", dict(lw=1, ls=":"))):
            ax.plot(*[v if i == 0 else 100 * v for i, v in enumerate(OG.running(y))], label=label, **kw)
        ax.set_xlabel("queries seen"); ax.set_ylabel("accuracy in each block of 500 queries (%)")
        ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2); fig.tight_layout()
        md = (f"**{len(got):,} held-out queries of the three benchmarks in random order.** The learner starts with no labels and, "
              + ("for every query, is told whether each of the two answers was correct" if both else
                 "for every query, is told only whether the answer it showed was correct")
              + f"; {eps:.0%} of its decisions are random. Accuracy over the whole stream **{got.mean():.1%}**, over the last "
              f"third **{got[-third:].mean():.1%}** (reads {act[-third:].mean():.0%} of queries). Last third of the same stream: "
              f"development-fitted gate {np.where(uf > 0, c1, c0)[order][-third:].mean():.1%}, always-retrieve "
              f"{c1[order][-third:].mean():.1%}, never-retrieve {c0[order][-third:].mean():.1%}. Simulated on logged outcomes; "
              "it assumes every shown answer is judged correctly.")
        return fig, md
    return sim


def byo(f):
    """Bring-your-own-log: frontiers of an error- and a benefit-trained gate fitted on the uploaded log, and its audit."""
    d = pd.read_csv(f.name if hasattr(f, "name") else f)
    feats = [c for c in d.select_dtypes("number") if c not in ("c_ne", "c_ev", "group", "segment", "cost_ne", "cost_ev", "decision", "conf_ne", "conf_ev")]
    cne = d["cost_ne"].to_numpy() if "cost_ne" in d else np.zeros(len(d))
    cev = d["cost_ev"].to_numpy() if "cost_ev" in d else np.ones(len(d))
    grp = d["group"].to_numpy() if "group" in d else np.arange(len(d))
    u = (d.c_ev - d.c_ne).to_numpy().astype(int)
    X = d[feats].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    out = {}
    rel = {}
    fitted = [("benefit gate", gate.oof_scores(X, u, grp, kind="benefit")), ("error gate", gate.oof_scores(X, (1 - d.c_ne).to_numpy(), grp))] if feats else []
    for name, sc in fitted + [("oracle", u.astype(float))]:
        acc, cost = gate.curve(sc, d.c_ne, d.c_ev, cne, cev)
        out[name] = gate.nauc(acc)
        rel[name] = gate.cost_at_accuracy(acc, cost, d.c_ev.mean() - 0.005) / cev.mean()
        ax.plot(gate.GRID, acc * 100, label=name, ls="--" if name == "oracle" else "-")
    ax.set_xlabel("fraction of queries that read evidence"); ax.set_ylabel("accuracy (%)"); ax.grid(alpha=.3)
    ax.legend(fontsize=8); fig.tight_layout()
    t = gate.outcome_table(d.c_ne, d.c_ev)
    shown = feats if len(feats) <= 12 else feats[:12] + [f"… ({len(feats)} in all)"]
    md = f"n={len(d)}, signals: {', '.join(shown) or 'none'} · retrieval helps {t['help']:.1%}, hurts {t['harm']:.1%}"
    if feats:
        md += (f" · nAUC benefit {out['benefit gate']:.4f} vs error {out['error gate']:.4f} "
               f"(oracle {out['oracle']:.4f}) · cost at always-retrieve accuracy (−0.5 pt), relative to always: "
               f"benefit {rel['benefit gate']:.2f}×, error {rel['error gate']:.2f}×"
               + ("" if "cost_ev" in d else " (no cost columns: unit cost per read assumed)"))
    else:
        md += " · no signal columns: no gate is fitted; the audit below covers the fixed policies, the logged gate and the oracle"
    conf = dict(conf_ne=d["conf_ne"].to_numpy(), conf_ev=d["conf_ev"].to_numpy()) if {"conf_ne", "conf_ev"} <= set(d.columns) else {}
    audit = (gate.audit(X, d.c_ne, d.c_ev, grp, d["segment"].to_numpy(), decision=d["decision"].to_numpy() if "decision" in d else None, n_boot=300, **conf)
             if "segment" in d else None)
    if audit is not None:  # one compact column for the difference and its interval
        audit["vs told [95% CI]"] = [f"{v:+.1f} [{lo:+.1f}, {hi:+.1f}]" for v, lo, hi in zip(audit.vs_fixed, audit.lo, audit.hi)]
        audit = audit.drop(columns=["vs_fixed", "lo", "hi"]).rename(columns={"accuracy": "acc.", "read": "read"})
        audit["policy"] = audit.policy.replace(AUDIT_NAMES)  # the names used in the paper
    sc_gate = gate.oof_scores(X, u, grp, kind="benefit") if feats else None
    few = "segment" in d and 1 < d["segment"].nunique() <= 20  # with a few large segments: one gate per segment in the planner's segment rows
    sc_seg = gate.oof_scores_by_segment(X, u, grp, d["segment"].to_numpy()) if feats and few else None
    # with many small segments (question types) the planner shows the log as a whole; the audit handles the segments
    plan = gate.plan(d.c_ne, d.c_ev, segment=d["segment"].to_numpy() if "segment" in d and d["segment"].nunique() <= 20 else None, score=sc_gate,
                     score_segment=sc_seg, n_boot=200)
    if not feats:
        return fig, md, audit, plan, None
    path = pathlib.Path(tempfile.mkdtemp()) / "vgate_benefit_gate.json"  # the gate fitted on the whole log, as plain weights
    path.write_text(json.dumps(gate.export_gate(X, d.c_ne, d.c_ev, feats), indent=1))
    return fig, md, audit, plan, str(path)


BYO_HELP = ("Upload a CSV with one row per query: `c_ne` (closed-book correct, 0/1), `c_ev` (correct with "
            "retrieval, 0/1), optional `group` (e.g. image id), optional `cost_ne` / `cost_ev` (tokens or "
            "seconds of each pass), and any numeric signal columns. VGate fits a benefit gate and an error "
            "gate (5-fold, grouped), draws both frontiers and reports the cost at always-retrieve accuracy. "
            "With a `segment` column (the source or type of each query) it also audits what the gate has "
            "learned: does it beat the better fixed policy of each segment, is it matched by a classifier "
            "that only predicts the segment, and does it carry over to a segment it was not fitted on? "
            "The segment can be fine-grained (a question type or template per query): the audit then tells you whether the gate "
            "has learned anything that holds for types it has not seen. "
            "A `decision` column (0/1: did your own gate read the evidence?) puts that gate through the same audit; "
            "`conf_ne` and `conf_ev` (a confidence of each reply; give 0 to a reply that cannot be used, e.g. one that names no option) "
            "add the rule that reads everything and answers with the more confident reply.")
BYO_AUDIT_LABEL = ("audit (acc. = accuracy, mean over segments; read = % of queries read; vs told = difference to the "
                   "told-the-segment policy with a 95% cluster-bootstrap interval); needs a `segment` column. With more than 20 "
                   "segments (e.g. question types): accuracy is the mean over queries, per-segment policies are estimated on the "
                   "training folds and the unseen-segment gates leave whole segments out")
AUDIT_NAMES = {"fixed per segment (needs the label)": "told the segment", "benefit gate per segment (needs the label)": "gate per segment", "segment predicted": "segment classifier",
               "after reading: the more confident answer": "after reading: more confident answer",
               "logged gate (decision column)": "logged gate", "logged gate: its majority decision per segment": "logged gate: segment majority"}


def byo_tab():
    """The bring-your-own-log view; returns nothing, wires its own events."""
    gr.Markdown(BYO_HELP)
    with gr.Row():
        log_file = gr.File(file_types=[".csv"], label="log CSV")
        sample = gr.Button("Use the sample log (held-out runs of the three benchmarks)", visible=SAMPLE_LOG.exists())
        published = gr.Button("Use a published system's log (Chen et al., GPT-4o, six sources; logged gate: their fine-tuned gate)", visible=PUBLISHED_LOG.exists())
    byo_plot = gr.Plot()
    byo_md = gr.Markdown()
    byo_audit = gr.Dataframe(label=BYO_AUDIT_LABEL)
    byo_plan = gr.Dataframe(label="should you gate? per segment: help and harm (%), the better fixed policy and the paired labels that find it (95%), "
                                  "oracle headroom and null_lead (what the best threshold of a score with NO signal would lead by on a log of this size, in points: "
                                  "the noise floor for a threshold tuned on the labels it is scored on; up to twice that if the score's sign is tuned too); the help-vs-harm AUROC a gate needs to lead that policy by one point (auroc_floor: no score gets "
                                  "there below it; auroc_needed: a score of the usual shape, with its 5-95% range for a log of this size); the AUROC of "
                                  "a benefit gate fitted on that segment of this log (out of fold; last row: on all of it), the most any score of that "
                                  "AUROC can lead by (ceiling), what a score of the usual shape leads by (estimate) and what the gate did lead by (lead), "
                                  "in points")
    byo_gate = gr.File(label="benefit gate fitted on the whole log (JSON weights; apply with vgate.gate.apply_gate, read iff the value is > 0)")
    log_file.upload(byo, log_file, [byo_plot, byo_md, byo_audit, byo_plan, byo_gate])
    sample.click(lambda: byo(str(SAMPLE_LOG)), None, [byo_plot, byo_md, byo_audit, byo_plan, byo_gate])
    published.click(lambda: byo(str(PUBLISHED_LOG)), None, [byo_plot, byo_md, byo_audit, byo_plan, byo_gate])


def log_only_ui(note="The case-study outputs are not on disk (see REPRODUCIBILITY.md), so only the log view is shown."):
    with gr.Blocks(title="VGate") as ui:
        gr.Markdown("## VGate — audit a retrieval gate from its log\n" + note)
        with gr.Tab("Audit your own log"):
            byo_tab()
        gr.Markdown(f"<small>{FOOTER}</small>")
    return ui


def build_ui(store, init_ds=None, init_qid=None, init_budget=0.4, init_auto=False, live=None, feedback=None):
    datasets = list(store.ds)
    if not datasets:  # no case-study outputs on disk (a fresh clone): the log view alone, which needs only a CSV
        return log_only_ui()
    GATE = store.gate
    with gr.Blocks(title="VGate") as ui:
        gr.Markdown("## VGate — retrieve only when it helps\nEach query: the VLM answers without evidence, the gate "
                    "predicts whether retrieved evidence would **help or hurt**, and evidence is read only when the "
                    "predicted benefit clears the budget threshold.")
        with gr.Row():
            ds = gr.Dropdown(datasets, value=init_ds or datasets[0], label="dataset")
            budget = gr.Slider(0, 1, value=init_budget, step=0.01, label="retrieval budget (fraction of queries)")
            auto = gr.Checkbox(value=init_auto, label="budget-free rule: read iff predicted benefit > 0")
        with gr.Tab("Inspector"):
            qid = gr.Dropdown(label="question", choices=[])
            with gr.Row():
                img = gr.Image(label="image", height=300)
                with gr.Column():
                    q_md = gr.Markdown()
                    decision = gr.Markdown()
            signals = gr.Dataframe(label="pre-read signals (no evidence read yet)")
            answers = gr.Dataframe(label="answers and cost")
            evidence = gr.Dataframe(label="retrieved evidence (top-5)")
        with gr.Tab("Frontier"):
            unit = gr.Radio(["tokens", "reads"], value="tokens", label="cost measured as")
            plot = gr.Plot()
            summary = gr.Markdown()
            flips = gr.Dataframe(label="decisions that flip between this budget and budget + 10 points")
        with gr.Tab("Harm auditor"):
            harm = gr.Dataframe(label="queries where always-retrieve broke a correct answer")
        if live is not None:
            with gr.Tab("Live"):
                gr.Markdown("Upload an image and ask a question. The model answers without evidence; VGate-pre decides "
                            "whether to search; if it does, the model writes a one-sentence caption, a dense retriever "
                            "searches Wikipedia passages with caption + question, and VGate-full decides whether the "
                            "model reads them. Budget-free rule: act only if the predicted benefit is positive.")
                with gr.Row():
                    l_img = gr.Image(label="your image", type="pil", height=300)
                    with gr.Column():
                        l_q = gr.Textbox(label="question")
                        l_all = gr.Checkbox(value=False, label="also run the passes VGate skips (to compare with always-retrieve)")
                        l_go = gr.Button("Ask")
                        l_dec = gr.Markdown()
                l_ans = gr.Dataframe(label="answers and cost")
                l_ev = gr.Dataframe(label="retrieved evidence (top-5)")
                l_t = gr.Markdown()

                def live_answer(image, question, run_all):
                    if image is None or not (question or "").strip():
                        return "Upload an image and type a question.", None, None, ""
                    return live_views(live(image, question.strip(), bool(run_all)))

                l_go.click(live_answer, [l_img, l_q, l_all], [l_dec, l_ans, l_ev, l_t])
        if feedback is not None:
            with gr.Tab("Learn from feedback"):
                gr.Markdown("A gate fitted with one retriever does not transfer to another, and fitting needs both answers "
                            "judged. Here the gate is learned during use: the system answers a stream of queries and is "
                            "told only whether the answer it showed was right.")
                with gr.Row():
                    f_eps = gr.Slider(0, 0.3, value=0.05, step=0.01, label="share of random decisions (exploration)")
                    f_both = gr.Checkbox(value=False, label="judge both answers (the paired-outcome reference)")
                    f_seed = gr.Number(value=0, precision=0, label="stream order (seed)")
                    f_go = gr.Button("Run the stream")
                f_plot = gr.Plot()
                f_md = gr.Markdown()
                f_go.click(feedback, [f_eps, f_both, f_seed], [f_plot, f_md])
        with gr.Tab("Audit your own log"):
            byo_tab()

        def qids_for(d):
            ch = store.qids(d)
            return gr.update(choices=ch, value=init_qid if init_qid in ch else ch[0])

        def inspect(d, q, b, au=False):
            S, r = store.ds[d], store.ds[d]["recs"][q]
            g, f = S["g"].loc[q], S["feats"].loc[q]
            thr = store.threshold(d, GATE, b, au)
            search = True
            if f"score_{PRE_GATE}" in g:
                search = g[f"score_{PRE_GATE}"] >= store.threshold(d, PRE_GATE, b, au)
            retrieve = bool(search and g[f"score_{GATE}"] >= thr)  # cascade: read only if stage 1 searches and stage 2 passes
            final = g.ev_text if retrieve else g.ne_text
            final_ok = g.c_ev if retrieve else g.c_ne
            md = f"**Q:** {r['question']}\n\n**Gold:** {r['gold_answer']}"
            credit = store.image_credit(r)
            if credit:
                md += f"\n\n<small>{credit}</small>"
            elif store.public:
                md += "\n\n<small>Image not shown: its source does not permit redistribution.</small>"
            pre_line = ""
            if f"score_{PRE_GATE}" in g:
                thr1 = store.threshold(d, PRE_GATE, b, au)
                pre_line = (f"**Stage 1 — search?** VGate-pre: predicted benefit {g[f'score_{PRE_GATE}']:+.3f} vs "
                            f"threshold {thr1:+.3f} → {'SEARCH' if search else 'SKIP SEARCH'}\n\n")
            dec = (f"### Decision: {'🔎 READ EVIDENCE' if retrieve else '✅ ANSWER DIRECTLY'}\n{pre_line}"
                   f"**Stage 2 — read?** {GATE_NAME[GATE]}: predicted benefit {g[f'score_{GATE}']:+.3f} vs threshold "
                   f"{thr:+.3f} " + ("(budget-free rule)" if au else f"at budget {b:.0%}") + "\n\n"
                   f"final answer: **{final}** — {'correct' if final_ok else 'wrong'}")
            sig = pd.DataFrame([{"signal": s, "value": round(float(f[s]), 3)} for s in SIGNALS if s in f and pd.notna(f[s])])
            gate_base = int(U.base_cost(S["g"].loc[[q]], S["pre"] + ["qpp"]).iloc[0])  # NE + feature passes
            ans = pd.DataFrame([
                {"policy": "Never retrieve", "answer": g.ne_text, "correct": int(g.c_ne), "tokens": int(g.tok_ne),
                 "GPU-s (T4)": round(float(g.gpu_ne), 2)},
                {"policy": "Always retrieve", "answer": g.ev_text, "correct": int(g.c_ev), "tokens": int(g.tok_ev),
                 "GPU-s (T4)": round(float(g.gpu_ev), 2)},
                {"policy": "VGate", "answer": final, "correct": int(final_ok),
                 "tokens": gate_base + (int(g.tok_ev) if retrieve else 0),
                 "GPU-s (T4)": round(float(g.gpu_ne) + (float(g.gpu_ev) if retrieve else 0), 2)}])
            ev = pd.DataFrame([{"rank": i + 1, "title": e["title"], "score": round(e["score"], 3), "source": wiki_url(e),
                                "text": e["text"][:300]} for i, e in enumerate(r["evidence"])])
            path = store.img.get(os.path.basename(r["image_file"])) if store.can_show(r) else None
            return path, md, dec, sig, ans, ev

        def frontier(d, b, u="tokens"):
            g = store.ds[d]["g"]
            t = gate.outcome_table(g.c_ne, g.c_ev)
            s = (f"never {t['acc_never']:.1%} · always {t['acc_always']:.1%} · oracle gate {t['acc_oracle_gate']:.1%} · "
                 f"retrieval helps {t['help']:.1%} of queries and **hurts {t['harm']:.1%}**")
            return store.frontier_fig(d, b, u), s

        def flip_table(d, b):
            S = store.ds[d]; g = S["g"]
            lo, hi = store.threshold(d, GATE, b), store.threshold(d, GATE, min(1.0, b + 0.10))
            f = g[(g[f"score_{GATE}"] >= hi) & (g[f"score_{GATE}"] < lo)]
            return pd.DataFrame([{"qid": q, "question": S["recs"][q]["question"], "without evidence": r.ne_text,
                                  "with evidence": r.ev_text, "retrieval": {1: "helps", 0: "no change", -1: "hurts"}[int(r.u)]}
                                 for q, r in f.head(30).iterrows()])

        def harm_table(d, b, au=False):
            S = store.ds[d]
            g = S["g"][S["g"].u == -1]
            thr = store.threshold(d, GATE, b, au)
            thr1 = store.threshold(d, PRE_GATE, b, au) if f"score_{PRE_GATE}" in g else -np.inf
            return pd.DataFrame([{"qid": q, "failure mode (heuristic)": failure_mode(row.ev_text, S["recs"][q]["evidence"]),
                                  "question": S["recs"][q]["question"], "gold": S["recs"][q]["gold_answer"],
                                  "no-evidence (right)": row.ne_text, "with evidence (wrong)": row.ev_text,
                                  "top evidence": S["recs"][q]["evidence"][0]["title"],
                                  "VGate reads?": bool(row.get(f"score_{PRE_GATE}", np.inf) >= thr1 and row[f"score_{GATE}"] >= thr)}
                                 for q, row in g.iterrows()])

        gr.Markdown(f"<small>{FOOTER}</small>")
        ds.change(qids_for, ds, qid)
        for trig in (qid.change, budget.release, auto.change):
            trig(inspect, [ds, qid, budget, auto], [img, q_md, decision, signals, answers, evidence])
        for trig in (ds.change, budget.release, unit.change, auto.change):
            trig(frontier, [ds, budget, unit], [plot, summary])
            trig(flip_table, [ds, budget], flips)
            trig(harm_table, [ds, budget, auto], harm)
        ui.load(qids_for, ds, qid).then(frontier, [ds, budget, unit], [plot, summary]).then(harm_table, [ds, budget, auto], harm)
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="experiments/runs/J4,experiments/runs/J6")  # held-out (confirm) outputs; comma-separated
    ap.add_argument("--model", default="Qwen3-VL-4B-Instruct")
    ap.add_argument("--split", default="confirm")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--init_ds", default=None)
    ap.add_argument("--init_qid", default=None)
    ap.add_argument("--init_budget", type=float, default=0.4)
    ap.add_argument("--init_auto", action="store_true", help="start with the budget-free rule switched on")
    ap.add_argument("--public", action="store_true", help="only show images that may be redistributed with credit "
                    "(Google Landmarks v2, CC BY); other images are withheld")
    ap.add_argument("--live", default="", help="folder with the passage index (passage_emb.npy, passages.jsonl): adds a "
                    "Live tab that runs the model on an uploaded image; needs a CUDA GPU (see vgate.live)")
    ap.add_argument("--live_dtype", default="float16", help="float16 on a T4; bfloat16 on Ampere cards such as the A2000")
    a = ap.parse_args()
    live = None
    if a.live:
        from vgate.live import GATES_DIR, Gates, LivePipeline, Retriever
        from vgate.models import VLM, prepare_image
        gates = GATES_DIR / f"{a.model}__live.npz"  # refit on logs of this retriever (PREREG v2g); the benchmark-fitted
        gates = gates if gates.exists() else GATES_DIR / f"{a.model}.npz"  # gates do not transfer to another retriever
        pipe = LivePipeline(VLM("Qwen/" + a.model, dtype=a.live_dtype), Gates(gates), Retriever(a.live))

        def live(image, question, run_all):
            res = pipe.answer(prepare_image(image), question, run_all=run_all)
            res.pop("hidden", None)
            return res
    build_ui(Store(a.run, a.model, a.split, public=a.public), a.init_ds, a.init_qid, a.init_budget, a.init_auto, live,
             feedback_sim(a.model, a.split)).launch(share=a.share, server_port=a.port)

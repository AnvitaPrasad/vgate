# VGate — auditing what retrieval gates learn

**Try it:** the log view runs in your browser at https://huggingface.co/spaces/AnvitaPrasad/vgate (nothing you upload leaves your
machine). **Raw per-query outputs** of the four readers: https://huggingface.co/datasets/AnvitaPrasad/vgate-outputs
(`python scripts/fetch_outputs.py`).
**Video** (2 min 31 s, captions, no sound): `demo/vgate_demo.mp4`, recorded from the application by `scripts/record_video.py`.

Research prototype and WSDM 2027 demo. Vision-language models answer knowledge-intensive visual questions better
with retrieved evidence, but retrieval costs searches and tokens and can overturn correct answers, so systems gate
it per query. VGate (1) fits retrieve-or-skip gates from a system's outcome logs, at two decision points (search?
read?), with a ledger of the tokens of every model call, and (2) **audits what a gate has learned**: any gate — ours,
or one whose 0/1 decisions are a column of a log — is compared with the better fixed policy of each traffic segment,
a classifier that only predicts the segment, an oracle gate, and itself on a segment it was not fitted on.

```
pip install -e ".[demo]"
python demo/app.py            # Inspector · Frontier · Harm auditor · Learn from feedback · Audit your own log (CPU)
python demo/app.py --live <folder with the passage index>   # adds the Live view (needs a CUDA GPU; see vgate.live)
```
A log for the audit is a CSV with `c_ne`, `c_ev` (is the closed-book / the retrieval-augmented answer correct),
any numeric signals, optional `group`, `cost_ne`, `cost_ev`, a `segment` column, and optionally `decision`.
`demo/sample_log.csv` is the held-out log of the main model. `demo/sample_log_published.csv` is another group's log: GPT-4o on the Mix set of Chen et al. (arXiv:2502.18023), scored from their released answers with their metric (correct above 0.5 token accuracy), with their fine-tuned gate's search decisions as the logged gate (`scripts/chen_logs_score.py`, `scripts/chen_logs_audit.py`; PREREG v11).

**Case study (held-out; four frozen VLMs, three benchmarks; `experiments/PREREG.md` says which results are
preregistered, registered or exploratory; every number is generated, see `paper/tables/numbers.tex`):**
- Retrieval changes the main model's accuracy by +28 / +10 / −20 points on
  E-VQA / InfoSeek / OK-VQA.
- In mixed traffic one gate gains 3.7–5.7 points over always-retrieve across the four
  models; a classifier that only predicts the benchmark is 0.3–0.5 points
  ahead of it, and being told the benchmark 0.8–1.1 ahead. On a single
  benchmark no gate of ours beats the better fixed policy by one point (exploratory).
- Fitted on two benchmarks and applied to the third, the best gate is 3.9–6.1
  points below the better fixed policy.
- Published gates through the same audit (preregistered; as released, nothing fitted): the search-need prompt of Chen et al.
  is −11.2 points from the told-the-benchmark policy, UniversalRAG's router prompt retrieves on at least
  96% of every benchmark (−6.5), and on development data the ReAG critic is −4.1.
- What a gate can earn (exploratory; `scripts/gate_bound.py`, `gate.plan`, the "should you gate?" table of the log view).
  A gate that reads a set of queries gains h·TPR − m·FPR over never-retrieve (h, m: shares of queries that retrieval
  helps / harms), so its lead over the better fixed policy is at most min(h, m)·(2A − 1) for a score with a concave
  ROC curve of help-vs-harm AUROC A (the ceiling), and about G*(h, m, A) for a score of binormal shape (the estimate;
  not a bound). On 156 held-out scores the ceiling is never exceeded and observed leads sit at 4% of it in the median;
  the estimate is within 0.08 points of the best lead on average; with the threshold chosen on other images the scores
  lead by −0.02 points on average. A one-point lead is out of reach below an AUROC of 0.58–0.64 and needs 0.81–0.91 at
  the estimate; the best score has 0.75. The mathematics is classical cost-sensitive ROC analysis (Provost and Fawcett
  2001; Drummond and Holte 2006); related accounting: TARG (arXiv 2511.09803), Bouchard (2605.06350), Bhatti and
  Vaddina (2610.01535), Hauff et al. (ECIR 2010).
- A gate per benchmark with more labels (exploratory; `scripts/tail_gate_check.py`, `scripts/tail_gate_diag.py`): fitted out of
  fold on each benchmark's held-out log, a gate leads the better fixed policy by a point in 2 of 24 fits, both the main reader's
  on InfoSeek (+1.2 logistic, +1.5 boosted trees). InfoSeek's held-out questions use 390 question strings; with whole strings
  held out its leads are −1.5 to +0.1 on the four readers. `gate.audit` has a mode for many small segments (question types).
- On a fourth benchmark where retrieval both helps (8.3%) and hurts (14.0%) the same reader (CVQA, held-out,
  preregistered), the gate leads the better fixed policy by +0.4 [−0.2, +1.0], not by the
  registered one point; an oracle gate would lead by 8.3.
  There, a rule that decides after reading (answer with the evidence reply only when it names an option and is the more
  probable at its first token; no fitted parameter; registered) leads that policy by +1.6 [+0.9, +2.4] at
  1.24× the tokens of always-retrieve, and the gate by +0.8 [+0.1, +1.4] like for like; on the open-ended benchmarks'
  development splits it does not help.
- Fitted on its own traffic, a benefit-trained read gate beats an error-trained one with identical features by
  1.64 nAUC points [1.26, 2.01] (preregistered).
- A gate learned from feedback on the shown answer alone ends within 0.8 points of the fitted one (simulation).
- What did not work is logged too: token savings, answer selection after reading, a Thinking checkpoint, passage
  selection with the reader's own check or a released critic, transfer across retrievers.

Start with `experiments/PREREG.md` (criteria and outcome logs) and
`REPRODUCIBILITY.md`.

```
src/vgate/        data · prompts · models (HF fp16/bf16, cost ledger) · run · evaluation (official scorers)
                  uncertainty (features) · gate (labels, out-of-fold gates, frontiers, bootstrap, audit) · live · cvqa
scripts/          score · analyze (dev) · confirm (held-out) · extend · operating_point (mixed traffic) · lodo
                  online_gate · external_gates · cvqa_audit · multiplicity · paper_numbers · kaggle.sh / push_kernel.sh
kaggle/<job>/     GPU jobs          demo/app.py      Gradio demo
experiments/      PREREG.md · results/ · audits/ · runs/ (git-ignored outputs)
paper/            acmart sigconf, 4 pages, all numbers generated
```

## Data and model credits
The MIT licence in `LICENSE` covers the code only. Data and model outputs follow the terms of their sources
(`data/README.md`).

Questions: Encyclopedic-VQA (Mensink et al., ICCV 2023), InfoSeek (Chen et al., EMNLP 2023; Apache-2.0) and OK-VQA
(Marino et al., CVPR 2019), accessed through M2KR (Lin et al., ACL 2024; MIT). Images are **not** redistributed here:
iNaturalist 2021 (non-commercial research use, no redistribution), Google Landmarks v2 (CC BY per image), OVEN, and
COCO val2014 (Flickr terms). Passages are Wikipedia text (CC BY-SA) via WikiWeb2M / WIT, EchoSight and M2KR. Models:
Qwen3-VL and InternVL3.5 (Apache-2.0), bge-base-en-v1.5 (MIT), BEM (Apache-2.0), Qwen2.5-VL-3B (Qwen Research License,
non-commercial). Built with Qwen.

## About this copy
This is the public release: code, demo, frozen splits, result files and the preregistration log. Working notes, the paper's source and the raw model outputs are not in it (the raw outputs are a separate download; see `REPRODUCIBILITY.md`). The hand audits list question id, verdict and note only. `experiments/PREREG.md` is the unedited log: it refers to internal check notes (`docs/experiments/reviewer_simulation_*.md`) that are not released.

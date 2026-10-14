# Preregistration: Gate 1 pilot (v0, 2026-09-30)

Frozen at a git commit **before any pilot output is scored**. v1 (frozen at Gate 1, Oct 3 18:00) may only
(a) prune features per the pruning rule and (b) fix the headline per the decision table. Nothing else changes.

## Samples
- Dev = image-grouped samples, seed 2027: `data/splits/{evqa,infoseek}_dev.txt` (1,000 each).
  sha256 in `data/splits/SPLITS.sha256`. Confirmatory = the disjoint remaining image groups (`*_confirm.txt`).
- Evidence (EV) = top-5 benchmark retrievals (E-VQA: PreFLMR sections; InfoSeek: EVA-CLIP pages → best
  120-word BM25 chunk per page vs the question). Oracle = dataset gold evidence (upper bound only).

## Labels
- c_without = correct(NE greedy), c_with = correct(EV greedy); **u = c_with − c_without ∈ {−1, 0, +1}**.
  help: u=+1; harm: u=−1. Greedy decoding, `max_new_tokens=32`, frozen prompts (`prompts.PROMPT_ID`),
  images capped at 512×512 area.
- Correctness: E-VQA = official protocol (EM, else BEM ≥ 0.5; PyTorch BEM port); InfoSeek = official
  scorer (String/Time EM over `answer_eval`, Numerical range with ±10% default).
- Error label for the comparison gate: e = 1 − c_without.

## Features (whitelist; `data.FEATURE_BLACKLIST` + tests/test_features.py forbid leakage)
Pre-search (NE-only): token entropy (mean/max/first), min & mean token logprob, sequence logprob, k=5 sample
agreement with the greedy answer + distinct-answer count, entity-name k=5 agreement, P(True), paraphrase
agreement, hidden state (last prompt token) at L/2, 3L/4, L−1 (probe baselines only), prompt length.
Post-search adds (scale-free, z-scored on training folds only): top1−top2 score margin / top1, softmax entropy
over top-10 scores, NQC-style std of top-10 scores.

## Gates and evaluation
- L2 logistic regression; GroupKFold(5) by image_id; thresholds chosen inside training folds.
- 2×2: {error label, benefit label} × {pre-search features, post-search features}; factorised P̂(wrong)·P̂(hit)
  with P̂(hit) trained on `lab_gold_in_top5` as a target only.
- Cost = prompt + generated tokens of every pass the policy executes (NE + feature passes + EV if taken);
  GPU-seconds reported as a secondary cost.
- **Primary endpoint:** paired ΔnAUC of the accuracy-vs-cost frontier (benefit gate − error gate, same
  features), cluster bootstrap by image (2,000 resamples). Secondary: help/harm AUROC (DeLong, Holm).

## Criteria (pilot CIs = 80%, cluster bootstrap)
Validity (fail → fix, not kill):
- V1 no NaN/inf in token logprobs; (on A2000 when available) fp16-vs-bf16 correctness agreement ≥ 95%.
- V2 Qwen2.5-VL-3B NE within ±3 pt of ReAG (E-VQA 21.9 BEM, InfoSeek 18.3); one prompt/pixel fix allowed.
- V3 u flip rate on a 100-q greedy rerun ≤ 3%; ≥ 60% of 50 audited harm cases are real (else use
  "stable harm": BEM ≥ 0.7 without and ≤ 0.3 with evidence).

C1 (Qwen3-VL-4B-Instruct, 2,000 pooled dev questions):
- C1-a headroom: always − never ≥ +5 pt on ≥ 1 dataset AND help rate ≥ 8%. Fail on both → stop.
- C1-b harm: pooled harm ≥ 2.5%. Else no harm claims.
- C1-c label effect: benefit gate beats error gate by ΔnAUC ≥ +0.75 pt (CI > 0) or saves ≥ 15% cost at
  always-retrieve accuracy. Pass = STRONG; fail = WEAK.
- C1-d practical: some gate within 0.5 pt of always-retrieve accuracy at ≤ 75% of its total cost on ≥ 1 dataset.
- Pruning: keep k-sample / entity features only if they add ≥ 0.02 pooled help-AUROC over free features.

C2 (Qwen3-VL-4B-Thinking vs Instruct):
- C2-0 median think length ≤ 2,000 tokens and truncation at 4,096 ≤ 10% (J0). Fail → C2 cut.
- C2-a unique help P(think ✓ ∧ NE ✗ ∧ EV ✗) − noise floor ≥ 3 pt.
- C2-c 3-way router beats best 2-action gate at matched GPU-seconds by ΔnAUC ≥ 1 pt.
- Promote C2 only if C2-0, C2-a, C2-c all pass.

## Decision table
| C1 | C2 | Headline |
|---|---|---|
| STRONG | promoted | "Answer, Think, or Search?" router |
| STRONG | not promoted | Two-stage (pre/post-search) retrieval gate |
| WEAK | promoted | C2 router |
| WEAK | not promoted | Gatekeeper + harm-auditor tool with its findings (replication of Qian et al. in VLMs) |

---

# PREREG v1 — frozen 2026-09-30 ~18:00 IST, before any confirmatory (confirm-split) output exists

## Gate-1 outcome (pilot = dev, 1,000 E-VQA + 1,000 InfoSeek, Qwen3-VL-4B-Instruct fp16)
Source: `experiments/runs/J1/analysis__Qwen3-VL-4B-Instruct__dev{,__pruned}.{json,md}`, `scripts/gate1_report.py`.
- V1 no NaN in any pass (fp16-vs-bf16 check pending A2000). V2 pending J2. V3: u flip rate on 100-q shuffled rerun
  = 1.0% (E-VQA) / 1.0% (InfoSeek) ≤ 3% → PASS; harm audit 30/50 (E-VQA, at threshold) and 44/50 (InfoSeek) real → PASS.
- C1-a PASS (E-VQA +26.2 pt, help 31.7%; InfoSeek +8.4 pt, help 13.6%). C1-b PASS (pooled harm 5.35%).
- **Pruning rule applied:** no paid feature group adds ≥ 0.02 pooled help-AUROC over free features (deltas: +sample
  −0.013, +entity −0.004, +P(True) +0.004, +para −0.001, +all −0.008) → **sample, entity, ptrue, para are dropped.**
- C1-c on the pruned feature sets: pre-search (free) +0.05 pt [−0.20, +0.30]; **post-search (free+QPP) +1.36 pt
  [+0.91, +1.78] → STRONG** (strict labels +1.41 [+0.97, +1.83]).
- C1-d FAIL (best 1.09× / 0.85× always-retrieve total tokens) → per v0 rule, **no token-cost savings claim**; cost
  results are reported descriptively, and savings are claimed only in search calls (always-acc − 0.5 pt reached at
  85% / 59% of searches on dev).
- C2: pending J3 (running); decided by the v0 C2 criteria. Headline per the v0 decision table.

## Confirmatory design (fixed now)
- Data: confirm image groups (`*_confirm.txt`: E-VQA 2,750, InfoSeek 3,637). Model 1: Qwen3-VL-4B-Instruct fp16
  (job J4: conditions ne, ev, oracle — no paid passes). Model 2: Qwen2.5-VL-3B-Instruct (J2 full-set ne/ev/oracle;
  its confirm rows).
- Features: free (NE token entropy/log-prob stats, answer length) and QPP (retriever score features). Gates:
  error_pre (free), benefit_pre (free), error_post, benefit_post, factorised_post (free+QPP); single-signal baselines;
  hidden-state probes; never / always / random / oracle.
- **Training: every gate is fit once on that dataset's full dev set (same model) and applied unchanged to its
  confirm set.** No refitting, thresholds, or feature choices on confirm data.
- **Primary endpoint:** pooled (mean over the two datasets) paired ΔnAUC(benefit_post − error_post) on confirm,
  95% image-cluster bootstrap CI (2,000 resamples). Confirmed if the CI excludes 0 and the point estimate > 0.
- Secondary (reported regardless of outcome): per-dataset ΔnAUC; benefit_post − factorised_post (two-sided);
  benefit_post − error_pre (vs. NE-only uncertainty gating); per-signal error/help/harm AUROC; help/harm rates;
  searches needed to reach always-retrieve accuracy − 0.5 pt; total-token cost curves (descriptive); strict-label
  sensitivity; model-2 replication; cross-dataset transfer (fit on dataset A dev, apply to dataset B confirm).

## Outcome log (appended after v1; criteria above are unchanged)
- 2026-09-30 19:55 — **V2 PASS.** Qwen2.5-VL-3B NE on full sets: E-VQA 22.9 [21.4, 24.5] vs ReAG 21.9; InfoSeek 16.7
  [15.5, 18.1] vs ReAG 18.3 (both within ±3). Oracle differs by design (gold section vs ReAG's page).
- 2026-09-30 20:00 — **Model-2 confirmatory (Qwen2.5-VL-3B, fit dev → test confirm, 2,750 + 3,637 q):** primary
  dnAUC(benefit_post − error_post) = +1.49 pt, 95% CI [+1.05, +1.87] → confirmed for model 2. benefit_pre − error_pre
  −0.01 [−0.26, +0.24]; benefit_post − factorised_post −0.20 [−0.48, +0.09]; benefit_post − error_pre +1.00
  [+0.61, +1.36]. Source: `experiments/runs/J2/confirm__Qwen2.5-VL-3B-Instruct.{json,md}`.
- 2026-09-30 21:45 — **Main-model confirmatory (Qwen3-VL-4B, fit dev → test confirm, 2,750 + 3,637 q): PRIMARY
  ENDPOINT CONFIRMED.** dnAUC(benefit_post − error_post) = +1.64 pt, 95% CI [+1.28, +2.00] (E-VQA +2.30 [+1.68, +2.89];
  InfoSeek +0.97 [+0.58, +1.36]); strict labels +1.58 [+1.23, +1.93]. Secondary: benefit_pre − error_pre −0.15
  [−0.39, +0.09]; benefit_post − error_pre +1.10 [+0.79, +1.40]; **benefit_post − factorised_post −0.35 [−0.58, −0.12]
  (factorised better)**; transfer: benefit > error in both directions. Held-out help 32.9% / 14.2%, harm 4.7% / 4.6%.
  Searches to reach always-acc − 0.5 pt: factorised 81% / 83%, benefit 90% / 95%; no gate below 1.07× always-retrieve
  tokens. Source: `experiments/runs/J4/confirm__Qwen3-VL-4B-Instruct{,__strict}.{json,md}`,
  `experiments/results/signal_auroc_confirm.csv`.
- 2026-09-30 22:45 — **Corrections after hostile-review simulation** (`docs/experiments/reviewer_simulation_1.md`):
  (i) bootstrap now resamples image groups *within each dataset* (stratified), matching the stated design (code had
  pooled); primary unchanged: Qwen3-VL-4B +1.64 [+1.26, +2.01], Qwen2.5-VL-3B +1.49 [+1.06, +1.89], strict +1.58
  [+1.21, +1.95]. (ii) the v1-listed secondary gates that confirm.py had omitted are now run: **hidden-state probes**
  (PCA-64 + logistic on the NE pass's last-prompt-token state, layer 3L/4 fixed a priori; fit on dev, applied to
  confirm): probe_benefit − probe_error = **+2.15 [+1.75, +2.56]** (Qwen3-VL-4B), **+2.17 [+1.74, +2.60]**
  (Qwen2.5-VL-3B); all three captured layers +1.73…+2.37 (exploratory robustness). Random policy added.
  (iii) single-signal NQC threshold (already a v1 single-signal baseline) now reported with CIs: benefit_post −
  NQC +0.23 [−0.05, +0.52]; NQC − error_post +1.41 [+0.89, +1.87]. (iv) **Correction of a reporting error:** the
  "fraction of queries that search" axis/metric counted evidence *reads*; post-search gates query the retriever on
  every query. Search savings are only attributable to pre-search gates. (v) Exploratory (not preregistered):
  partial AUC over 70–100% reads, benefit_post − error_post +0.81 [+0.45, +1.18] / +0.62 [+0.23, +1.00].
- 2026-10-01 01:40 — **C2 (Thinking vs Instruct, dev, 500 + 500 q): NOT PROMOTED.** C2-0 PASS (median think length
  548 / 293 tokens; truncation 0.6% / 0%). C2-a FAIL: unique help (think ✓ ∧ NE ✗ ∧ EV ✗) minus the noise floor
  (Instruct sampled at the Thinking settings) = +1.6 pt [+0.6, +2.8] (E-VQA) / +2.6 [+1.2, +4.0] (InfoSeek), pooled 2.1
  < 3 pt. C2-c therefore not evaluated. Descriptive: thinking fixes 10% / 8% of NE-wrong answers (retrieval 47% /
  16%), breaks 5.4% / 6.2% of all questions, costs 38.5 / 27.2 GPU-s per query vs 0.9 s for the EV pass; two thinking
  samples agree on 92% / 94.5% of labels. Headline per the decision table: two-stage retrieval gate.
  Source: `experiments/runs/J3/` (c2_summary.json, scores__*.csv).
- 2026-10-01 02:40 — Second review pass (`docs/experiments/reviewer_simulation_2.md`). The cascade was evaluated at
  budgets fixed before looking (0.9 and 0.5; `experiments/runs/J4/cascade__*.json`). The thinking break-rate
  denominator was corrected (NE-correct subset). No analysis choice was changed after seeing these numbers.
- 2026-10-03 — **Added replication, declared before any output exists:** model 3 = InternVL3.5-4B-HF (different
  family). Same protocol as model 2: ne and ev on all questions (job J5); gates fit on dev and tested on confirm with
  `scripts/confirm.py`. Endpoints reported: primary (benefit_post − error_post) and probe (probe_benefit −
  probe_error), 95% stratified cluster bootstrap. No oracle or paid features. Reported regardless of outcome.
- 2026-10-03 — J5 v1 (InternVL3.5-4B-HF) failed with CUDA OOM before producing any output: InternVLProcessor tiles
  images into up to 13 × 256 tokens by default. Fix (infrastructure, not analysis): `crop_to_patches=False`, i.e. a
  single 448-px tile of 256 image tokens, comparable to the Qwen models' image budget. Rerun as J5 v2.
- 2026-10-04 — **Model-3 replication (InternVL3.5-4B-HF, J5 v2, fit on dev → test on confirm): replicated.**
  - Primary: +1.78 pt [+1.40, +2.17].
  - Probe: +2.33 [+1.89, +2.76].
  - Scalar pre-search: −0.20 [−0.46, +0.07].
  - benefit_post − factorised_post: −0.25 [−0.45, −0.04].
  - benefit_post − NQC: +0.06 [−0.19, +0.30].
  - Accuracy: E-VQA NE 20.6 → EV 53.4; InfoSeek 9.5 → 19.3 (all questions).
  - Source: `experiments/runs/J5/confirm__InternVL3_5-4B-HF.{json,md}`.

# PREREG v2 addendum — OK-VQA extension (2026-10-04, before any OK-VQA model output exists)
Motivation: test a regime where always-retrieve is reported to hurt strong VLMs (commonsense knowledge).
- **Data:** M2KR `OKVQA_data` test (5,046 questions, COCO val2014 images from `lmms-lab/OK-VQA`).
  - Image-grouped split, seed 2027: dev = image groups up to ≥ 1,000 questions; confirm = the rest.
- **Retrieval:** dense `BAAI/bge-base-en-v1.5` over the 114,809 M2KR OK-VQA passages.
  - Query = dataset-provided image caption + question; no VLM pass is used for the query.
  - Evidence = top-5 passages (title + text, word-capped as for the other datasets); QPP features = the top-20 scores.
  - `lab_gold_in_top5` = any M2KR pseudo-relevant `pos_item_ids` in the top-5 (factorised baseline target only).
- **Correctness:** official VQA soft accuracy (10 annotators, leave-one-out average, VQA answer normalisation).
  - Binary label c = 1 iff soft accuracy ≥ 0.6 (≥ 2 annotators agree), fixed now. Soft accuracy is reported descriptively.
- **Models:** Qwen3-VL-4B-Instruct (main) and Qwen2.5-VL-3B-Instruct; conditions ne and ev on all questions.
- **Analysis:** identical to v1. Gates are fit on dev and applied unchanged to confirm.
  - Primary (benefit_post − error_post) and probe (probe_benefit − probe_error), 95% stratified cluster bootstrap.
- **H-OK (headline for this dataset):** VGate-post, at the read rate that maximises its out-of-fold dev accuracy
  (chosen on dev only), achieves higher held-out accuracy than **both** never-retrieve and always-retrieve. Each
  difference gets a 95% paired cluster-bootstrap CI.
- Kill/report rule: reported whatever the outcome. If always-retrieve does not hurt on OK-VQA, H-OK is reported as
  not supported.

# PREREG v2b addendum — new components (2026-10-04, before any OK-VQA model output exists)
Status: **confirmatory on OK-VQA** (no outputs exist yet). **Exploratory on E-VQA and InfoSeek**, whose held-out
results for other gates have already been seen; there, everything is fit on dev only and reported as exploratory.
Retrieval for OK-VQA is computed by `kaggle/okvqa-retrieval` (the same code as `scripts/build_okvqa_retrieval.py`,
max_seq_length 256).

1. **VGate-full (post-search gate):** the benefit model (multinomial over u) on [free NE features, QPP features,
   PCA-64 of the layer-3L/4 hidden state].
   - Endpoints: ΔnAUC vs VGate-post (benefit_post) and vs VGate-pre (probe_benefit), 95% stratified cluster bootstrap.
2. **VGate-select (answer selection after reading):** both passes are run and the system returns a_E or a_0.
   - Selector: the same multinomial benefit model on [free NE features, QPP, the same token statistics of the EV
     pass (entropy mean/max/first, log-prob min/mean/sum, length), 1{norm(a_0) = norm(a_E)}, 1{a_E abstains}].
   - Rule: return a_E iff score ≥ τ, where score = P̂(u=+1) − P̂(u=−1) and τ maximises out-of-fold dev accuracy.
     Model and τ are fit on dev only.
   - **H-SEL:** held-out accuracy(VGate-select) − accuracy(always-retrieve) > 0, 95% paired cluster bootstrap.
     Also reported: the difference to never-retrieve, the fraction of answers switched, and the precision of switches.
   - Cost is reported honestly: NE + EV passes for every query, i.e. more than always-retrieve.
3. **Four-outcome probe (MMA-RAG-style baseline, our reimplementation of the idea):** PCA-64 + logistic over
   {both wrong, retrieval fixes, retrieval breaks, both right} on the hidden state, ranked by P̂(fixes) − P̂(breaks).
   Reported next to VGate-pre.
4. No other analysis choice changes. Results are reported whatever the outcome.

# PREREG v2c addendum — model-4 replication, Qwen3-VL-8B-Instruct (2026-10-04, before any 8B output exists)
Purpose: test whether the benefit-vs-error effect holds for a model twice the size of the main one.
- **Job:** `kaggle/j7-q3vl8b`. fp16 + sdpa, transformers 5.16.1, the same prompts, image cap and decoding as every
  other model. The weights do not fit one T4, so one process holds the model split across both GPUs
  (`--device auto`); nothing else differs. Conditions `ne`, `ev` on the frozen dev and held-out splits of E-VQA and
  InfoSeek.
- **Analysis:** `scripts/confirm.py` and `scripts/extend.py` at the commit that contains this addendum, unchanged.
  Gates are fit on dev and applied to the held-out split, as for models 2 and 3.
- **Replication criteria (same as models 2 and 3):** pooled ΔnAUC(benefit_post − error_post) with its 95% stratified
  cluster-bootstrap CI above 0, and pooled ΔnAUC(probe_benefit − probe_error) with its CI above 0. The hidden-state
  layer is 3L/4, as before.
- **Also reported (same definitions as v2b):** VGate-full vs benefit_post / probe_benefit / factorised_post, the
  four-outcome probe vs probe_benefit, and VGate-select vs always-retrieve.
- **Run status rules:** a crashed, blocked or timed-out job is FAILED / BLOCKED / incomplete, never a negative
  result. If only part of the held-out split finishes, it is reported as partial with its n and not pooled with
  complete runs. Results are reported whatever the outcome.
- Optional, quota permitting: the same two conditions on OK-VQA (v2), analysed as in v2 / v2b.

# Outcome log, v2b / v2c (appended; criteria above unchanged)
- 2026-10-04 — **Selector screening on dev only** (no held-out data): out-of-fold dev accuracy of the registered
  logistic selector vs variants (difference features, +hidden state, gradient boosting) on Qwen3-VL-4B and
  Qwen2.5-VL-3B. No variant was consistently better (all within ±0.5 pt), so the registered selector is kept.
- 2026-10-04 — **v2b components on E-VQA / InfoSeek: EXPLORATORY** (added after these held-out sets had been
  used for other gates; everything fit on dev, `scripts/extend.py`, 2,000 bootstrap replicates).
  Source: `experiments/runs/{J4,J2,J5}/extend__<model>.{json,md}`.
  - **Qwen3-VL-4B-Instruct** (pooled ΔnAUC, pt): VGate-full − VGate-post +0.75 [+0.40, +1.08]; − VGate-pre +0.40 [+0.20, +0.60]; − factorised +0.39 [+0.03, +0.76]; − error gate +2.38 [+1.91, +2.83]. Benefit probe − four-outcome probe -0.02 [-0.08, +0.06]; four-outcome probe − error probe +2.17 [+1.76, +2.57].
    - evqa: VGate-select 51.5 vs always 51.2 (Δ +0.29 [-0.08, +0.68]; oracle selection 55.9; 6.2% switched, precision 62%). VGate-post at its dev-chosen read rate (99%) − always +0.07 [-0.08, +0.20].
    - infoseek: VGate-select 27.9 vs always 27.4 (Δ +0.58 [+0.27, +0.91]; oracle selection 31.9; 7.7% switched, precision 82%). VGate-post at its dev-chosen read rate (89%) − always -0.85 [-1.33, -0.45].
  - **Qwen2.5-VL-3B-Instruct** (pooled ΔnAUC, pt): VGate-full − VGate-post +0.80 [+0.47, +1.13]; − VGate-pre +0.36 [+0.17, +0.52]; − factorised +0.60 [+0.23, +0.99]; − error gate +2.29 [+1.82, +2.72]. Benefit probe − four-outcome probe +0.03 [-0.04, +0.09]; four-outcome probe − error probe +2.14 [+1.71, +2.57].
    - evqa: VGate-select 48.0 vs always 48.6 (Δ -0.62 [-1.20, +0.00]; oracle selection 53.3; 11.9% switched, precision 38%). VGate-post at its dev-chosen read rate (92%) − always +0.00 [-0.42, +0.45].
    - infoseek: VGate-select 26.8 vs always 26.6 (Δ +0.25 [+0.00, +0.49]; oracle selection 31.0; 2.4% switched, precision 71%). VGate-post at its dev-chosen read rate (100%) − always +0.00 [+0.00, +0.00].
  - **InternVL3_5-4B-HF** (pooled ΔnAUC, pt): VGate-full − VGate-post +0.50 [+0.16, +0.84]; − VGate-pre +0.54 [+0.37, +0.71]; − factorised +0.25 [-0.08, +0.61]; − error gate +2.28 [+1.83, +2.75]. Benefit probe − four-outcome probe -0.10 [-0.18, -0.03]; four-outcome probe − error probe +2.43 [+1.98, +2.88].
    - evqa: VGate-select 53.7 vs always 53.6 (Δ +0.04 [+0.00, +0.12]; oracle selection 57.7; 0.4% switched, precision 100%). VGate-post at its dev-chosen read rate (99%) − always -0.07 [-0.26, +0.11].
    - infoseek: VGate-select 19.6 vs always 19.6 (Δ +0.05 [-0.48, +0.57]; oracle selection 23.1; 30.2% switched, precision 51%). VGate-post at its dev-chosen read rate (99%) − always +0.00 [-0.21, +0.14].
  - Reading: pooled over the two datasets, VGate-full beats VGate-post and VGate-pre on all three models and is
    at least on par with the factorised baseline, which needs gold-retrieval labels (per dataset it is not always
    best, e.g. InternVL on E-VQA: 41.8 vs VGate-post 42.1). The four-outcome probe and the benefit probe are
    equivalent (|Δ| ≤ 0.10 pt). VGate-select does **not** reliably beat always-retrieve (−0.62 to +0.58 pt;
    CI above 0 on 1 of 6 model-dataset pairs). On these entity-centric benchmarks the accuracy-optimal read
    rate is ≈ 90–100%, so H-OK-style accuracy gains are not expected here; OK-VQA is the confirmatory test.
- 2026-10-04 — J7 v1 (Qwen3-VL-8B) produced no model output: the launch command put `timeout` before the
  environment variables (rc 127). Infrastructure fix only; rerun as J7 v2.
- 2026-10-04 — OK-VQA retrieval built by `kaggle/okvqa-retrieval` (cuda); `data/raw/okvqa_retrieval.parquet`
  sha256 d2c574d7…c41cd4. Table and splits built (dev 1,000 / confirm 4,046; no image overlap); pseudo-relevant
  passage in top-1 / top-5: 6.0% / 13.9%. J6 was pushed at ~13:42 IST; this entry was written while it was
  still running, before any OK-VQA model output could be read (Kaggle exposes outputs only on completion).

# PREREG v2d addendum — selection in MMA-RAG's setting (2026-10-04, before J6 finished; no OK-VQA output read)
Why: re-reading MMA-RAG (arXiv 2603.00511v1) showed that its classifier decides **after** retrieval, from hidden
states of both the closed-book pass and the retrieval-augmented pass (H_c = Concat(T1, V1, T2, V2)); retrieval always
runs. Its setting is answer selection (our VGate-select), not a pre-search gate. Our "four-outcome probe" (v2b) applies
its target to the closed-book hidden state only and must be described that way.
- **Data:** job `kaggle/j8-evhidden` re-runs the greedy `ev` pass of Qwen3-VL-4B-Instruct on E-VQA, InfoSeek and
  OK-VQA with `--hidden_conds ev` (last prompt token, layer 3L/4), same splits, order and batch size as J1 / J4 / J6.
  Labels and answers stay those of the original runs; agreement of the regenerated answers is reported.
- **Selector-H (MMA-RAG setting, our reimplementation with a linear classifier, not their trained model):**
  multinomial logistic over the four outcome classes on [PCA-64(h_NE), PCA-64(h_EV)], each standardised and fit on dev;
  score = P̂(fixes) − P̂(breaks); return a_E iff score ≥ τ, with τ maximising out-of-fold dev accuracy.
- **Selector-HF:** the same with VGate-select's features added (NE and EV token statistics, QPP, agreement, abstention).
- **Endpoints (Qwen3-VL-4B):** held-out accuracy of each selector minus always-retrieve, and minus VGate-select,
  95% paired image-cluster bootstrap, per dataset. Cost is reported as NE + EV passes for every query.
- **Status:** exploratory on E-VQA and InfoSeek. On OK-VQA the design is fixed here before any OK-VQA output was read,
  but the hidden states are produced after the OK-VQA answers will have been scored, so it is reported as a
  pre-specified secondary analysis, not as a confirmatory test. Reported whatever the outcome.

# Outcome log — OK-VQA (PREREG v2 / v2b), 2026-10-04 ~15:20 IST
Run J6 (`kaggle/j6-okvqa`, build fc206451eba7): ne + ev on all 5,046 questions, no NaN, no empty answers.
Source: `experiments/runs/J6/{confirm,extend}__<model>.{json,md}`. Fit on dev (1,000), tested on confirm (4,046).
- **Qwen3-VL-4B-Instruct**: never 59.6, always 39.9, oracle gate 64.5; retrieval helps 4.9% and harms 24.5% of queries.
  - Primary (benefit_post − error_post): +0.33 [-0.13, +0.76] → NOT CONFIRMED (CI includes 0).
  - Probe (probe_benefit − probe_error): +1.55 [+1.06, +2.02] → CONFIRMED. Scalar uncertainty (benefit_pre − error_pre): +0.73 [+0.34, +1.15].
  - **H-OK: NOT SUPPORTED.** The dev-chosen read rate of VGate-post is 0%, i.e. the gate collapses to never-retrieve: − never +0.00 [+0.00, +0.00], − always +19.65 [+18.13, +21.23].
  - **H-SEL: supported as stated, but trivially.** VGate-select 59.6 vs always 39.9 (+19.65 [+18.13, +21.23]); vs never +0.00 [+0.00, +0.00]: it returns a_0 for nearly every query where the answers differ, so it only recovers never-retrieve.
  - VGate-full − VGate-post +1.37 [+0.85, +1.88]; − VGate-pre +0.67 [+0.44, +0.91]; − factorised +1.93 [+1.33, +2.50]; − error gate +1.69 [+1.12, +2.23]. Benefit probe − four-outcome probe +0.05 [-0.04, +0.15].
- **Qwen2.5-VL-3B-Instruct**: never 59.5, always 45.9, oracle gate 65.2; retrieval helps 5.6% and harms 19.3% of queries.
  - Primary (benefit_post − error_post): +0.82 [+0.25, +1.37] → CONFIRMED.
  - Probe (probe_benefit − probe_error): +1.21 [+0.66, +1.75] → CONFIRMED. Scalar uncertainty (benefit_pre − error_pre): +0.94 [+0.38, +1.49].
  - **H-OK: NOT SUPPORTED.** The dev-chosen read rate of VGate-post is 3%, i.e. the gate collapses to never-retrieve: − never -0.02 [-0.27, +0.20], − always +13.62 [+12.19, +15.09].
  - **H-SEL: supported as stated, but trivially.** VGate-select 59.7 vs always 45.9 (+13.79 [+12.38, +15.28]); vs never +0.15 [-0.12, +0.42]: it returns a_0 for nearly every query where the answers differ, so it only recovers never-retrieve.
  - VGate-full − VGate-post +0.20 [-0.24, +0.66]; − VGate-pre +0.51 [+0.24, +0.79]; − factorised +1.00 [+0.44, +1.55]; − error gate +1.02 [+0.43, +1.58]. Benefit probe − four-outcome probe -0.05 [-0.10, +0.01].
- **Reading.** With this retriever (bge over Wikipedia passages, caption + question as query; pseudo-relevant
  passage in the top-5 for 13.9% of questions) retrieval hurts OK-VQA under the official metric, and no gate
  improves on never-retrieve at its dev-chosen operating point. The benefit-vs-error effect replicates for the
  hidden-state probe on both models and for the primary endpoint on Qwen2.5-VL-3B only.
- **Caveat found by reading harm cases (dev, 30 sampled, seed 2027; not a preregistered audit):** about 12 of 30
  are answer-form artefacts of exact-match VQA accuracy ("double" → "doubles", "cake" → "chocolate cake",
  "transport" → "transportation"); about 18 are genuinely worse answers (wrong entity copied from a passage,
  abstention, citation markers such as "[2]"). A containment-relaxed sensitivity label will be reported.

# PREREG v2e — operating-point analyses (2026-10-04, EXPLORATORY; declared before computing any of them)
Motivation: the OK-VQA outcome shows that the right read rate differs by dataset (≈ 0% on OK-VQA, ≈ 90–100% on
E-VQA / InfoSeek). A budget-free rule is needed to compare gates as they would be deployed.
1. **Natural-threshold rules** (models fit on dev exactly as in confirm.py / extend.py, no tuning):
   - benefit rule: read iff P̂(u=+1) − P̂(u=−1) > 0 (VGate-post and VGate-full);
   - error rule: read iff P̂(a_0 wrong) > 0.5 (the error gate with VGate-post's features);
   - references: never, always, and the better of the two as chosen on dev.
   Reported per dataset and macro-averaged over E-VQA, InfoSeek and OK-VQA: held-out accuracy and read rate;
   paired 95% image-cluster bootstrap (stratified by dataset) for benefit rule − error rule, benefit rule −
   always, benefit rule − never.
2. **Mixed traffic:** one gate fit on the union of the three dev sets with no dataset-identity feature (feature
   scaling and hidden-state PCA fit on the pooled dev set), applied to the union of the three held-out sets with
   the same rules. Same endpoints, macro-averaged over datasets, plus the per-dataset read rate.
3. **Containment-relaxed OK-VQA label (sensitivity only):** c = 1 if the official label is 1, or if an answer given
   by ≥ 2 annotators contains, or is contained in, the normalised prediction (both ≥ 3 characters). Applied to
   both passes; never used for a headline number.
Models: Qwen3-VL-4B-Instruct (main) and Qwen2.5-VL-3B-Instruct. All of this is post hoc and is labelled so.
- 2026-10-04 — v2e addition **made after seeing the first v2e results**: a robustness check with dev-tuned thresholds
  (`error_tuned`, `benefit_full_tuned` in `scripts/operating_point.py`: the threshold that maximises out-of-fold dev
  accuracy, macro-averaged over datasets), to test whether the error rule's natural threshold 0.5 is a weak baseline.
- 2026-10-04 — second v2e addition after seeing results: `error_rule_full` / `error_full_tuned`, the error target with
  VGate-full's features, so that the mixed-traffic comparison is equal-feature as in the primary endpoint.

# Outcome log — v2e operating points (EXPLORATORY), 2026-10-04 ~15:30 IST
Source: `experiments/results/operating_point__<model>[__relaxed].{json,md}` (`scripts/operating_point.py`).
- **Qwen3-VL-4B-Instruct, per-dataset gates** — macro accuracy over E-VQA, InfoSeek, OK-VQA: never 33.5, always 39.5, error rule 44.2 (71% reads), error rule with VGate-full's features 44.2 (68% reads), VGate-post rule 45.9 (65% reads), VGate-full rule 45.9 (59% reads), VGate-pre rule 45.9 (60% reads), oracle 50.8.
  - VGate-full rule − always +6.44 [+5.85, +6.99]; − never +12.47 [+11.41, +13.44]; − error rule (same features) +1.78 [+1.35, +2.22]; with dev-tuned thresholds +0.18 [-0.01, +0.38]. VGate-post rule − error rule +1.67 [+1.28, +2.06].
- **Qwen3-VL-4B-Instruct, mixed traffic (one gate)** — macro accuracy over E-VQA, InfoSeek, OK-VQA: never 33.5, always 39.5, error rule 42.2 (70% reads), error rule with VGate-full's features 43.5 (70% reads), VGate-post rule 42.9 (58% reads), VGate-full rule 45.2 (60% reads), VGate-pre rule 45.1 (62% reads), oracle 50.8.
  - VGate-full rule − always +5.67 [+5.09, +6.25]; − never +11.71 [+10.60, +12.68]; − error rule (same features) +1.69 [+1.21, +2.12]; with dev-tuned thresholds +1.23 [+0.74, +1.67]. VGate-post rule − error rule +0.69 [+0.18, +1.17].
  - Read rates E-VQA / InfoSeek / OK-VQA: VGate-full 84% / 82% / 12%; VGate-pre 90% / 84% / 10%; error rule (same features) 86% / 89% / 35%.
- **Qwen2.5-VL-3B-Instruct, per-dataset gates** — macro accuracy over E-VQA, InfoSeek, OK-VQA: never 32.9, always 40.4, error rule 43.6 (70% reads), error rule with VGate-full's features 43.6 (68% reads), VGate-post rule 45.0 (66% reads), VGate-full rule 44.3 (61% reads), VGate-pre rule 44.5 (62% reads), oracle 49.8.
  - VGate-full rule − always +3.95 [+3.44, +4.48]; − never +11.37 [+10.36, +12.36]; − error rule (same features) +0.73 [+0.31, +1.11]; with dev-tuned thresholds -0.08 [-0.25, +0.08]. VGate-post rule − error rule +1.41 [+1.05, +1.76].
- **Qwen2.5-VL-3B-Instruct, mixed traffic (one gate)** — macro accuracy over E-VQA, InfoSeek, OK-VQA: never 32.9, always 40.4, error rule 42.6 (72% reads), error rule with VGate-full's features 43.2 (71% reads), VGate-post rule 42.7 (66% reads), VGate-full rule 44.1 (63% reads), VGate-pre rule 44.0 (65% reads), oracle 49.8.
  - VGate-full rule − always +3.70 [+3.20, +4.24]; − never +11.12 [+10.08, +12.13]; − error rule (same features) +0.85 [+0.43, +1.29]; with dev-tuned thresholds +0.98 [+0.55, +1.40]. VGate-post rule − error rule +0.08 [-0.31, +0.46].
  - Read rates E-VQA / InfoSeek / OK-VQA: VGate-full 85% / 88% / 16%; VGate-pre 90% / 90% / 16%; error rule (same features) 88% / 89% / 35%.
- **Qwen3-VL-4B-Instruct [relaxed OK-VQA label], per-dataset gates** — macro accuracy over E-VQA, InfoSeek, OK-VQA: never 37.7, always 45.5, error rule 49.2 (67% reads), error rule with VGate-full's features 49.3 (63% reads), VGate-post rule 50.0 (65% reads), VGate-full rule 50.2 (60% reads), VGate-pre rule 50.2 (61% reads), oracle 55.4.
  - VGate-full rule − always +4.65 [+4.13, +5.17]; − never +12.53 [+11.47, +13.53]; − error rule (same features) +0.90 [+0.49, +1.29]; with dev-tuned thresholds +0.21 [+0.02, +0.41]. VGate-post rule − error rule +0.84 [+0.51, +1.16].
- **Qwen3-VL-4B-Instruct [relaxed OK-VQA label], mixed traffic (one gate)** — macro accuracy over E-VQA, InfoSeek, OK-VQA: never 37.7, always 45.5, error rule 47.2 (63% reads), error rule with VGate-full's features 48.9 (65% reads), VGate-post rule 47.5 (62% reads), VGate-full rule 49.4 (61% reads), VGate-pre rule 49.6 (63% reads), oracle 55.4.
  - VGate-full rule − always +3.85 [+3.32, +4.39]; − never +11.73 [+10.68, +12.70]; − error rule (same features) +0.50 [+0.10, +0.90]; with dev-tuned thresholds +0.78 [+0.35, +1.19]. VGate-post rule − error rule +0.27 [-0.15, +0.68].
  - Read rates E-VQA / InfoSeek / OK-VQA: VGate-full 86% / 84% / 13%; VGate-pre 92% / 85% / 13%; error rule (same features) 85% / 90% / 20%.
- **Reading.** Whether retrieval pays differs by regime (+28 / +10 / −20 points for Qwen3-VL-4B). A benefit gate
  with the budget-free rule "read iff predicted benefit > 0" adapts its read rate to the regime; an error gate
  cannot, because the same P(wrong) implies a different expected gain in each regime. Per dataset, the benefit
  rule matches the better of never / always without tuning (a per-dataset tuned threshold on either gate does the
  same). On mixed traffic, one benefit gate beats always, never and the equal-feature error gate on both models,
  also with dev-tuned thresholds and under the relaxed OK-VQA label (smaller margins).
- Containment-relaxed OK-VQA label (all 5,046 questions): Qwen3-VL-4B never 72.1 / always 57.4 (official 59.6 /
  39.7), harm 20.6% (24.7%), help 5.9% (4.8%); Qwen2.5-VL-3B never 71.2 / always 60.0, harm 17.6%, help 6.5%.
  Retrieval still hurts by about 15 / 11 points, so most of the OK-VQA harm is not a metric artefact.
- 2026-10-04 — third v2e addition, prompted by an independent review (`docs/experiments/reviewer_simulation_3.md`):
  reference policies `fixed_by_dataset` (told the dataset: the better of never / always on that dataset's dev split),
  `router` (predicts the dataset from the closed-book hidden state, then applies that policy) and `cascade` (search iff
  VGate-pre's benefit > 0, then read iff VGate-full's benefit > 0), plus the per-dataset breakdown of the margin over
  the error gate. Purpose: separate regime detection from per-query gating. Declared before computing them here.

# Outcome log — v2e reference policies (EXPLORATORY), 2026-10-04 ~16:10 IST
Prompted by `docs/experiments/reviewer_simulation_3.md`; source `experiments/results/operating_point__*.{json,md}`.
- **Qwen3-VL-4B-Instruct, mixed traffic** — macro accuracy: told the dataset 46.0; dataset router 45.6 (dataset predicted with 91.1% accuracy); VGate-full rule 45.2; VGate-pre rule 45.1; cascade 45.0 (62% searches, 56% reads); VGate-post rule 42.9; error rule with VGate-full's features 43.5.
  - VGate-full − router -0.44 [-0.81, -0.08]; − told-the-dataset -0.88 [-1.26, -0.53]. Margin over the equal-feature error gate by dataset: evqa -0.40 [-1.26, +0.42]; infoseek +0.47 [-0.06, +1.00]; okvqa +4.99 [+4.13, +5.86].
- **Qwen2.5-VL-3B-Instruct, mixed traffic** — macro accuracy: told the dataset 44.9; dataset router 44.5 (dataset predicted with 92.6% accuracy); VGate-full rule 44.1; VGate-pre rule 44.0; cascade 43.9 (65% searches, 60% reads); VGate-post rule 42.7; error rule with VGate-full's features 43.2.
  - VGate-full − router -0.45 [-0.78, -0.12]; − told-the-dataset -0.85 [-1.18, -0.51]. Margin over the equal-feature error gate by dataset: evqa -0.22 [-1.01, +0.52]; infoseek -0.27 [-0.73, +0.19]; okvqa +3.04 [+2.10, +3.98].
- **Qwen3-VL-4B-Instruct [relaxed OK-VQA label], mixed traffic** — macro accuracy: told the dataset 50.2; dataset router 49.9 (dataset predicted with 91.1% accuracy); VGate-full rule 49.4; VGate-pre rule 49.6; cascade 49.4 (63% searches, 58% reads); VGate-post rule 47.5; error rule with VGate-full's features 48.9.
  - VGate-full − router -0.48 [-0.86, -0.14]; − told-the-dataset -0.85 [-1.24, -0.50]. Margin over the equal-feature error gate by dataset: evqa -0.04 [-0.83, +0.76]; infoseek +0.38 [-0.19, +0.94]; okvqa +1.16 [+0.44, +1.90].
- **Reading (replaces the stronger wording of the first v2e entry).** The reviewer's point is confirmed on both
  models: in mixed traffic the benefit gate mostly detects the regime. A policy that is told the dataset, or one
  that predicts it from the closed-book hidden state, is 0.4–0.9 points *more* accurate than VGate-full, and the
  whole margin over the equal-feature error gate comes from OK-VQA. What remains true: from logged outcomes alone
  (no dataset labels, no budget) one benefit gate beats always-retrieve, never-retrieve and the equal-feature
  error gate. What is not achieved: per-query gains beyond the regime (oracle 50.8 vs 45–46).
- Paper changes made in response: Table 3 shows the dataset-aware rows and VGate-post; the abstract and the
  contribution list mark the mixed-traffic result as exploratory and state the gap to dataset-aware policies; the
  hidden-state comparison is described as preregistered for the replications only; the OK-VQA failure of the
  primary endpoint on the main model is in the abstract; audits are described as labelled with an LLM assistant.

# Outcome log — model-4 replication, Qwen3-VL-8B-Instruct (PREREG v2c), 2026-10-04 ~18:20 IST
Run J7 v2 (`kaggle/j7-q3vl8b`, build fc206451eba7): all four splits finished at the first attempt, no NaN, no
empty answers; fp16, model split over both T4s. Source: `experiments/runs/J7/{confirm,extend}__Qwen3-VL-8B-Instruct.{json,md}`.
- Held-out accuracy: E-VQA never 26.5 / always 54.8; InfoSeek 19.3 / 31.6.
- **Primary (benefit_post − error_post): +1.63 [+1.23, +1.99] → REPLICATED.**
- **Probe (probe_benefit − probe_error): +1.92 [+1.52, +2.32] → REPLICATED.**
- Scalar uncertainty (benefit_pre − error_pre): -0.09 [-0.30, +0.12] (tie, as on the other models).
- benefit_post − factorised_post -0.70 [-0.95, -0.44] (the factorised baseline is better); benefit_post − NQC -0.17 [-0.50, +0.16]; benefit_post − error_pre +0.99 [+0.67, +1.30].
- v2b components: VGate-full − VGate-post +0.94 [+0.63, +1.25]; − VGate-pre +0.49 [+0.32, +0.67]; − factorised +0.24 [-0.11, +0.59]; − error gate +2.57 [+2.10, +3.03]. Benefit probe − four-outcome probe -0.04 [-0.11, +0.02]. VGate-select never switches (dev-chosen threshold keeps a_E everywhere): +0.00 [+0.00, +0.00] / +0.00 [+0.00, +0.00].
- OK-VQA was not run on this model (it was optional in v2c; quota was kept for J8).
- 2026-10-04 — OK-VQA retrieval quality, analysis only: answer-string pseudo recall@5 of the bge retrieval is 59% on the
  held-out split (id-match with M2KR `pos_item_ids`: 13.9%). Literature check (`docs/strengthening/R3_okvqa_regime.md`):
  forced retrieval also hurts zero-shot Qwen2-VL on this corpus with PreFLMR (60.45 → 46.99, MMKB-RAG Table 6).

# PREREG v2f — system check of the live pipeline (2026-10-04, before job J9 is pushed)
This is a check of an engineering artefact, not a hypothesis test. Purpose: before the paper or the video says
anything about a live mode, establish that the live code path reproduces the logged runs and measure how it behaves.
- **Pipeline** (`src/vgate/live.py`): closed-book pass with the layer-3L/4 hidden state → VGate-pre → caption pass
  (the VLM describes the image in one sentence) → `BAAI/bge-base-en-v1.5` over the 114,809 M2KR OK-VQA passages, query
  = caption + question → VGate-full → evidence pass over the top-5. Gates: the mixed-traffic gates of v2e, frozen by
  `scripts/freeze_gates.py` (array implementation equal to scikit-learn within 1.1e-6; held-out macro accuracy equal
  to `operating_point__Qwen3-VL-4B-Instruct.json`).
- **Known differences from the logged runs, declared now:** (i) one query at a time instead of batches; (ii) for
  OK-VQA the logged retrieval used the dataset's captions, the live one uses a VLM-generated caption; (iii) for E-VQA
  and InfoSeek the live retriever and corpus are not the benchmark's (PreFLMR / EVA-CLIP over their own knowledge
  bases), so the read gate sees retriever-score features from a retriever it was not fitted on for those regimes;
  (iv) the passage matrix is stored in float16.
- **Job J9** (`kaggle/j9-live`, T4): builds the index, then runs every pass for every query on seeded samples of the
  held-out splits (seed 2027): 400 OK-VQA, 200 E-VQA, 200 InfoSeek.
- **Reported, whatever the outcome** (`scripts/live_report.py`, written before the outputs are read):
  1. agreement of the regenerated closed-book answers with the logged ones (normalised string match), and of
     u_pre recomputed live with the frozen-gate u_pre on the logged hidden states (correlation, sign agreement);
  2. OK-VQA only: overlap of the live top-5 with the logged top-5, and answer-string pseudo recall@5 of both;
  3. per dataset: accuracy of never-retrieve, always-retrieve (with the live retriever) and VGate under the
     budget-free rule, search and read rates, with 95% image-cluster bootstrap intervals;
  4. per-stage wall time (median, 90th percentile) and tokens.
- **Use:** if (1) shows sign agreement below 95%, the live path is not equivalent to the evaluated one and is fixed
  before anything else. (3) describes the live system; it does not replace or extend the paper's preregistered or
  exploratory results, which use the logged runs.

# Outcome log — live-pipeline system check (PREREG v2f), 2026-10-04 ~20:10 IST
Run J9 (`kaggle/j9-live`, build 63bce7ece0a4): index built in 24 min (114,809 × 768, float16); 800 held-out queries,
every pass executed, one query at a time. Source: `experiments/results/live_check.{json,md}` (`scripts/live_report.py`).
- **1. Equivalence with the logged runs: PASSED.** Closed-book answers equal the logged ones for 99.8% / 99.0% / 100.0% of OK-VQA / E-VQA / InfoSeek queries; u_pre recomputed live vs the frozen gate on logged hidden states: correlation 1.0000 / 1.0000 / 1.0000, sign agreement 100% / 100% / 100%.
- **2. OK-VQA retrieval with a VLM-written caption:** top-5 overlap with the dataset-caption retrieval 38%; answer-string pseudo recall@5 61% (live) vs 60% (logged): the caption source does not change retrieval quality.
- **3. Accuracy with the live retriever (bge over the OK-VQA passage set), budget-free rule, transferred gates:**
  - okvqa (n=400): never 58.0 [53.2, 63.0]; always 38.0 [33.0, 43.0]; VGate 57.8 [53.0, 62.7]; VGate − always +19.8 [+15.0, +25.0]; VGate − never -0.2 [-1.5, +1.0]; searches 8%, reads 7%. (Logged, benchmark retriever, same queries: never 58.0, always 38.8.)
  - evqa (n=200): never 22.5 [16.8, 28.4]; always 15.0 [10.4, 20.0]; VGate 19.0 [14.0, 24.5]; VGate − always +4.0 [+1.5, +7.0]; VGate − never -3.5 [-8.4, +1.1]; searches 90%, reads 80%. (Logged, benchmark retriever, same queries: never 23.0, always 51.5.)
  - infoseek (n=200): never 19.0 [13.9, 24.5]; always 11.5 [7.5, 16.0]; VGate 13.0 [8.6, 17.9]; VGate − always +1.5 [-0.5, +3.9]; VGate − never -6.0 [-10.3, -2.0]; searches 87%, reads 78%. (Logged, benchmark retriever, same queries: never 19.0, always 26.0.)
- **4. Latency on a T4, one query at a time (median):** closed-book 0.37–0.42 s, caption 1.6–1.9 s, retrieval 0.11 s
  (query encoder on CPU), evidence pass 1.15–1.33 s; about 3.5 s when every pass runs.
- **Reading.** The live code path is faithful to the evaluated one (item 1), so the gates behave live exactly as in
  the logged analysis. But the *system* is unsound for entity questions: with this retriever and corpus, retrieval
  hurts on E-VQA and InfoSeek too (15.0 vs 22.5; 11.5 vs 19.0), while the gates, fitted where the benchmarks'
  own retrievers made retrieval help, still search 87–90% and read 78–80% of those queries. VGate is then better
  than always-retrieve but worse than never-retrieve (−3.5 [−8.4, +1.1] and −6.0 [−10.3, −2.0]). This is the
  difference declared as (iii) in v2f: a benefit gate is specific to the model **and the retriever** it was fitted
  with. Consequence: no live mode is claimed in the paper with these gates; gates for a live system must be fit
  on outcomes logged with that system's retriever (the bring-your-own-log workflow).

# PREREG v2g — refitting the gates on logs of the live system (2026-10-04, declared before computing; no new GPU run)
Question: does the adoption workflow the paper describes (log both passes on a sample of traffic, fit, deploy) repair
the transferred gates' failure under the live retriever?
- **Data:** the 800 J9 queries (all passes logged): live a_0 and a_E scored with the official metrics, live hidden
  states, live retriever scores. The token statistics of the closed-book pass were not logged by J9; they are taken
  from the logged NE runs of the same queries (identical model and prompt; answers agree for 99–100%).
- **Refit gates:** the same two models and hyperparameters as `fit_benefit_gates` (VGate-pre: PCA of the hidden
  state, C=0.1; VGate-full: [free + QPP, PCA], C=1), one gate for the pooled traffic, PCA dimension min(64, n_train−1).
  Evaluation is out-of-fold: 5 folds grouped by image and stratified by dataset as far as GroupKFold allows; a query
  is always scored by a gate that never saw it.
- **Reported:** per dataset and macro: accuracy of never, always (live retriever), the transferred gates (v2f) and the
  refit gates under the budget-free rule; search and read rates; paired 95% image-cluster bootstrap intervals for
  refit − transferred, refit − never, refit − always.
- **Stated expectation:** with this retriever never-retrieve is the better fixed policy in all three regimes, so the
  refit gates should mostly decline to search; the open question is whether they recover the transferred gates' loss
  and whether any reads help. Reported whatever the outcome; exploratory; n is small (400 / 200 / 200).

# Outcome log — gates refit on the live system's logs (PREREG v2g, exploratory), 2026-10-04 ~20:20 IST
Source: `experiments/results/live_check.{json,md}` (section "Gates refit…"), `scripts/live_report.py`; 800 J9 queries, 5-fold out of fold.
- With the live retriever, retrieval helps / harms okvqa 5.8% / 25.8%, evqa 4.5% / 12.0%, infoseek 2.0% / 9.5%.
- Macro accuracy: never 33.2, always 21.5, transferred 29.9, refit 33.2, refit_pre_only 32.3.
- Read rates (OK-VQA / E-VQA / InfoSeek): transferred 7% / 80% / 78%; refit 13% / 15% / 10%.
- refit - transferred +3.25 [+1.25, +5.34]; refit - never +0.00 [-1.00, +1.01]; refit - always +11.67 [+8.92, +14.49]; transferred - never -3.25 [-5.44, -1.12]; refit_pre_only - never -0.83 [-2.02, +0.34].
- **Reading.** The stated expectation held: with this retriever the best fixed policy is never-retrieve everywhere, and
  the refit gates return to it (refit − never ±1.0) while the transferred gates lose 3.25 points. Refitting on a few
  hundred logged queries of the deployed system is therefore necessary and sufficient here; no query-level gain over
  never-retrieve was found. The gates refit on all 800 logs are frozen as `src/vgate/gates/Qwen3-VL-4B-Instruct__live.npz`
  and are what the Live tab loads.
- 2026-10-04 20:15 — J8 (`j8-evhidden`, PREREG v2d) is still running after 5 h against a 2 h estimate; its live log
  cannot be read (Kaggle returns HTTP 500). It is left to finish or to hit the platform's 12-hour limit; if it fails,
  v2d is recorded as NOT_RUN / FAILED, not as a negative result.

# PREREG v3 — PILOT of passage-level benefit selection (2026-10-04 ~22:45 IST, before job J10 is pushed)
**Status: pilot on development data. Nothing here supports a confirmatory claim.** A go decision leads to a
separate confirmatory registration (v3c) before any held-out output of the new conditions exists.

**Motivation (dev-only headroom analysis, recorded in `docs/STRENGTHENING.md`, part 2):** when the gold passage is in
the top-5, Qwen3-VL-4B is right 74.0% (E-VQA) / 33.9% (InfoSeek) of the time reading all five passages but 91.2% /
46.7% reading the gold passage alone; a perfect read/skip gate gains nothing, a perfect passage selector about +9.5 /
+9.8 points over always-retrieve.

**Conditions (job `kaggle/j10-passage-pilot`, Qwen3-VL-4B-Instruct, fp16, the three dev splits, prompts in
`src/vgate/prompts.py`, id c52a123f):**
- `ev1_k`, k = 1..5: the evidence prompt with retrieved passage k alone (word cap 300);
- `pv_k`, k = 1..5: P(yes) to "Is this context about what the image shows, and does it contain the answer to the question?";
- `ev_keep`: the top-5 evidence prompt that also states the closed-book answer and asks to keep it unless the context
  is clearly about the image and supports a different answer (a single-pass alternative);
- `ne` again (needed by `ev_keep`; expected to equal the logged closed-book pass).

**Candidates and labels.** Per question six candidate answers: a_0 (closed-book) and a_k from `ev1_k`; c_k by the
official metric (OK-VQA: soft accuracy ≥ 0.6).

**Selectors (fit and evaluated on dev by 5-fold cross-validation grouped by image; `scripts/passage_select.py`,
written before the outputs are read):**
- **VGate-pick (benefit target):** multinomial logistic on candidate rows (k = 1..5), target u_k = c_k − c_0, score
  P̂(u_k=+1) − P̂(u_k=−1); answer a_k* for the best-scoring passage if its score > 0, else a_0.
- **Correctness-target counterpart:** binary logistic P̂(c_k = 1) over the six candidates (same features; for k = 0
  the passage features are zero and an indicator is set); answer the arg-max candidate.
- **Features (per candidate):** token entropy / log-prob statistics and length of a_k; abstention flag; agreement of
  a_k with a_0; number of other passages giving the same answer; P(yes) from `pv_k` and its rank; retriever rank and
  z-scored retriever score of passage k; the closed-book pass's token statistics. No gold, hit or oracle field.
- **Ablations:** without the `pv` features (is the verification pass worth its cost?); without the answer statistics.
- **Baselines without training:** never; always (logged top-5 pass); top-1 passage only; arg-max P(yes) with a_0 when
  max P(yes) < 0.5; most confident non-abstaining passage answer (mean token log-prob); majority vote over the five
  passage answers; `ev_keep`; and the oracle over the six candidates.

**Reported:** accuracy per dataset with 80% image-cluster bootstrap intervals (pilot convention), paired differences
to always- and never-retrieve, and tokens per query (all passes a policy needs).

**Go / no-go (fixed now):** GO if VGate-pick beats always-retrieve by at least 3.0 accuracy points on E-VQA **and**
on InfoSeek with the 80% interval above 0, **and** is not more than 1.0 point below never-retrieve on OK-VQA.
Otherwise NO-GO: the result is logged, the method is not added to the paper as a positive result, and the paper stays
as it is. A crashed or partial job is not a negative result.

**Overlap guard:** results are framed as passage selection under a benefit target; no claim about how evidence pieces
combine or complement each other.

# PREREG v3a — loss decomposition on the held-out groups (descriptive; 2026-10-05 00:19 IST, commit `dbacb91`, before any held-out number of it is computed)
**Why.** A dev-only look at the logged runs (no new model pass) showed where the value of retrieval is lost. On E-VQA,
when the gold section is among the five passages and is not truncated by the 160-word cap, accuracy reading all five
depends on where the gold passage sits: rank 1 vs ranks 2–5 gives 88.6 vs 63.2 (Qwen3-VL-4B), 78.0 vs 61.3
(Qwen2.5-VL-3B), 86.5 vs 66.9 (InternVL3.5-4B), 89.8 vs 70.6 (Qwen3-VL-8B), while the gold passage alone gives 87–92
at either rank. Truncation costs a further ~13 points. On InfoSeek the retrieved 120-word chunk of the right page
contains an answer string in only 51% of the cases. These are development numbers and support no claim.

**Analysis (`scripts/loss_decomposition.py`, no fitted parameter, gold labels used for analysis only, never as a
feature).** On the held-out image groups, per dataset and model:
- never, always, oracle choice between the two answers, and the *ideal read/skip gate* (read iff the gold item is in
  the top-5);
- gold item in top-1 / top-5; accuracy reading all five given gold in top-5; accuracy with the gold passage alone
  (models that have the `oracle` condition: Qwen3-VL-4B, Qwen2.5-VL-3B);
- E-VQA only, gold in top-5 and not truncated: accuracy reading all five with the gold passage at rank 1 and at ranks
  2–5, and their difference (four models); the same split by closed-book correctness;
- E-VQA: accuracy reading all five for truncated vs untruncated gold passages;
- InfoSeek: share of right-page cases whose retrieved chunks contain an answer string, and accuracy by that flag.
Intervals: 95% image-cluster bootstrap (2,000 resamples).

**Expectations stated in advance (each is reported as held or not held):**
- D1: the ideal read/skip gate is within ±2.0 accuracy points of always-retrieve on E-VQA and on InfoSeek, for both
  models with a full set of conditions.
- D2: on E-VQA, rank-1 minus ranks-2–5 accuracy (untruncated, gold in top-5) is ≥ 10 points with the 95% interval
  above 0 for each of the four models.
- D3: on E-VQA, gold-alone minus all-five accuracy at ranks 2–5 is ≥ 10 points with the interval above 0 (two models).
**Caveat fixed now:** the rank of the gold passage is not randomised; a wrong passage ranked first by the retriever is
also the strongest distractor, so D2 does not separate order from distractor strength. No causal wording.

# PREREG v3 addendum — exploratory analyses of the pilot outputs (declared 2026-10-05 00:19 IST, commit `dbacb91`, before J10 outputs exist)
None of these changes the go / no-go rule of v3. All are development-data, exploratory.
- **E1 (which answer wins).** Share of questions whose top-5 answer equals the answer from passage k alone, k = 1..5,
  by the rank of the gold passage (gold label for analysis only).
- **E2 (reading-budget allocator).** A benefit-trained three-way choice among the closed-book answer, the top-1
  passage alone (`ev1_1`, 300 words) and the logged top-5 pass, from closed-book and retriever-score features only
  (no pass beyond the chosen read); accuracy, tokens, and the oracle over the three.
- **E3 (early exit).** Read passages in retriever order, stop at the first with a positive predicted benefit, with a
  cap of 2, 3 or 5 passages; accuracy against tokens.
- **E4 (expectation).** On InfoSeek the per-passage selector has little room, because the retrieved chunk lacks the
  answer in about half of the right-page cases. If the go rule fails on InfoSeek it is reported as a failure.

## Outcome log — v3a (held-out loss decomposition; computed 2026-10-05 00:30 IST, script commit `ffbbb20`, `experiments/results/loss_decomposition__confirm.{json,md}`)
Script committed with its dev output (which reproduces the dev table in `docs/STRENGTHENING.md`) before the held-out run.
- **D1 held.** Ideal read/skip gate − always-retrieve: Qwen3-VL-4B −0.5 [−1.4, 0.4] (E-VQA), +0.5 [−0.1, 1.1] (InfoSeek);
  Qwen2.5-VL-3B −1.5 [−2.5, −0.6], 0.0 [−0.6, 0.6]. (Not part of D1, same quantity: InternVL3.5-4B −2.7 / −0.5; Qwen3-VL-8B −1.0 / 0.0.)
- **D2 held.** E-VQA, gold in top-5 and untruncated (n = 1,145; 723 at rank 1, 422 at ranks 2–5), rank 1 − ranks 2–5:
  19.8 [14.2, 25.2] (Qwen3-VL-4B), 17.9 [11.9, 24.0] (Qwen2.5-VL-3B), 21.3 [15.1, 27.4] (InternVL3.5-4B), 18.2 [12.8, 24.0] (Qwen3-VL-8B).
  Split by the closed-book answer: 24.3 / 23.9 / 23.9 / 21.8 when it was wrong, 7.1 / −0.6 / 9.9 / 9.0 when it was right.
- **D3 held.** Gold passage alone − all five, gold at ranks 2–5: 29.6 [25.0, 34.3] (Qwen3-VL-4B), 27.7 [22.4, 32.9] (Qwen2.5-VL-3B);
  the gold passage alone scores 91.4 / 91.7 (rank 1 / ranks 2–5) and 88.0 / 86.0.
- Also reported: gold alone − all five over all gold-in-top-5 cases +19.9 [17.6, 22.6] / +20.1 (E-VQA) and +13.1 / +15.0 (InfoSeek);
  E-VQA untruncated vs truncated gold passage 74.6 vs 60.5 (Qwen3-VL-4B); InfoSeek: an answer string is inside the five
  retrieved chunks in 52.7% of right-page cases, accuracy 54.5 with it and 11.0 without (Qwen3-VL-4B).
- **Reading, with the registered caveat.** Rank is not randomised. Liu et al. (arXiv 2606.16494, checked 2026-10-05) ran the
  controlled version on the same retriever and benchmarks and report a first-vs-last gap of 16–26 points, but moving the
  gold passage to the first slot gained only +3.15 points (n = 254, not separable). So reordering is not the fix; the
  29-point gap to the gold passage alone is what a selection method could recover. On InfoSeek the gold-alone condition
  reads a different text (the page's passage) from the retrieved chunk, so that gap is chunk choice, not distraction.

# PREREG v3b — learning the gate from one-sided feedback (simulation on logged held-out runs; 2026-10-05 00:34 IST, commit `ee0439a`, before any held-out number of it is computed)
**Why.** The benefit label needs both answers judged, and gates fitted with one retriever do not transfer to another
(v2f). In deployment only the answer that was shown gets feedback. A dev-only simulation (pooled dev stream of 3,000
questions, 20 orders) gave 43.9% accuracy in the last third with one-sided feedback and 43.9% with both outcomes
judged (always 39.1, never 34.5). Development numbers; no claim.

**Simulation (`scripts/online_gate.py`, Qwen3-VL-4B, no new model pass).** Stream = the pooled held-out questions of
E-VQA, InfoSeek and OK-VQA (runs J4, J6) in random order, 10 orders (seeds 0–9). Inputs = VGate-full's inputs (free
and retriever-score features, 64 principal components of the closed-book hidden state); their standardisation and the
PCA are the frozen dev ones, which use no label. The learner starts with no labels: two L2 logistic models
(C = 0.05), P(correct | read) and P(correct | skip), each refit every 100 questions on the questions where that
answer was shown; it reads iff P̂(read) > P̂(skip); the first 100 questions and a share ε = 0.05 of the rest are random.
Feedback = correctness of the shown answer only.

**Comparators on the same streams:** never; always; the better fixed policy per dataset (told the dataset); the
dev-fitted VGate-full gate with the budget-free rule (both outcomes, frozen); the same online learner with both
outcomes judged; the oracle.

**Reported:** accuracy over the whole stream and over its last third, mean and range over the 10 orders (order
variation only; the stream is one sample, so no sampling interval is claimed), and the share of questions read.

**Expectations stated in advance:**
- O1: last-third accuracy with one-sided feedback is at least 3.0 points above always-retrieve in every order.
- O2: it is within 1.5 points of the online learner that sees both outcomes (mean over orders).
- O3: it is within 2.0 points of the dev-fitted frozen gate (mean over orders).
**Exploratory (not an expectation):** the same learner on the 800 logged live-retriever queries (J9) from a cold start,
against the transferred frozen gate and never-retrieve.
**Wording:** "one-sided feedback"; no treatment-effect vocabulary (overlap guard).

## Outcome log — v3b (one-sided feedback, held-out simulation; computed 2026-10-05 00:36 IST, `experiments/results/online_gate__confirm.{json,md}`)
Stream of 10,433 held-out questions, 10 orders; micro accuracy: never 35.4, always 38.5, better fixed policy per dataset 46.1, oracle 50.9.
- Online learner, shown answer judged only (ε = 0.05): whole stream 44.3 (44.1–44.5), last third **45.0** (44.0–45.7), reads 56%.
- Online learner, both outcomes judged: 44.8, last third 45.2 (44.2–46.3). Dev-fitted frozen VGate-full: 45.3, last third 45.3.
- **O1 held** (≥ +3.0 over always-retrieve in every order; last-third always 38.7). **O2 held** (0.2 below the
  both-outcomes learner). **O3 held** (0.3 below the frozen dev gate).
- Exploratory, the 800 live-retriever queries (cold start, micro): never 39.4, always 25.6, dev-fitted gate transferred
  36.6, online one-sided 37.0 over the whole stream (35.2–38.2) and 38.5 (35.0–40.5) over its second half.
- Reading: the gate needs neither paired outcomes nor a labelled set; feedback on the shown answer reaches the same
  accuracy within the stream. It is still the regime-level gain (below the per-dataset fixed policy, 46.1), and the
  simulation assumes every shown answer is judged correctly.

## Notes added after an independent claim check (2026-10-05 ~04:20 IST; `docs/experiments/reviewer_simulation_5.md`)
- **v3a, post-hoc addition.** The gold-alone prompt shows up to 400 words and the five-passage prompt 160 per passage, so
  the all-cases gap of 19.9 points mixes distraction with truncation. `scripts/loss_decomposition.py` now also reports the
  gap on the untruncated subset (same passage text in both prompts, n = 1,145): 16.9 [14.5, 19.6] (Qwen3-VL-4B) and
  17.6 [14.9, 20.6] (Qwen2.5-VL-3B). This quantity was added **after** the held-out run; it is a weighted combination
  of the registered rank-1 and ranks-2–5 rows. No earlier value changed (checked key by key). The paper quotes the
  registered ranks-2–5 gap (29.6) and this overall gap, the latter marked as computed after the registered run.
- **v3a, scope of D1.** D1 covered the two models with the gold-only condition. For the other two the same quantity is
  −2.7 [−3.7, −1.8] (InternVL3.5-4B, E-VQA; outside the ±2.0 band and below zero) and −1.0 [−2.0, 0.0] (Qwen3-VL-8B); on
  Qwen2.5-VL-3B E-VQA it is −1.5 [−2.5, −0.6]. The paper now says the gate gains nothing and is below zero on E-VQA for
  two of four models. One listed quantity (gold evidence for every question, J4) existed before the registration; no
  expectation rests on it.
- **v3b, the "Why" line.** Its dev numbers (43.9 / 43.9) came from an ad-hoc simulation run before the script was written
  (PCA-32 fitted on the stream, refit every 50 questions, ε = 0.1, 20 orders), not from `scripts/online_gate.py`; the
  committed script on the dev stream gives 44.9 / 44.9 over 3 orders (`experiments/results/online_gate__dev.md`). The
  expectations O1–O3 and the held-out outcome are unaffected.
- **v3b, live-retriever logs (exploratory).** The learner stays below never-retrieve there (37.0 over the stream, 38.5 over
  its second half, vs 39.4; transferred gates 36.6). The paper scopes the claim to the benchmark stream and does not
  report this number.

# PREREG v3b-rep — the one-sided-feedback simulation on a second model (2026-10-05 04:22 IST, commit `89f4e3b`, before any number of it is computed)
Same simulation, settings, comparators and expectations O1–O3 as v3b, for Qwen2.5-VL-3B-Instruct (runs J2 for E-VQA and
InfoSeek, J6 for OK-VQA; gates frozen by `scripts/freeze_gates.py --model Qwen2.5-VL-3B-Instruct --runs
evqa=J2:J2,infoseek=J2:J2,okvqa=J6:J6`, which self-checks against `operating_point__Qwen2.5-VL-3B-Instruct.json`).
`scripts/online_gate.py` gains a `--model` argument and nothing else changes. Reported whether or not the expectations hold.

## Outcome log — v3b-rep (Qwen2.5-VL-3B; computed 2026-10-05 04:25 IST, `experiments/results/online_gate__confirm__Qwen2.5-VL-3B-Instruct.{json,md}`)
Stream of 10,433 held-out questions, 10 orders; micro accuracy: never 34.9, always 39.9, better fixed policy per dataset 45.2.
Online learner with one-sided feedback: 43.4 over the stream, **44.2** (43.3–45.2) over the last third; with both outcomes 44.4;
dev-fitted frozen gate 44.6; always-retrieve 40.2 (last third). **O1, O2, O3 held.** The Qwen3-VL-4B numbers were regenerated
with the changed script and are identical (44.3 / 45.0; 44.8 / 45.2; 45.3 / 45.3).

# Outcome log — PREREG v3 pilot of passage-level benefit selection (J10; analysed 2026-10-05 ~09:15 IST)
Job `j10-passage-pilot` (build `0ebdb068cac2`) ran 2.54 h and wrote all 48 files (3 × 1,000 dev questions, 12 conditions; no
error, no NaN; the closed-book re-run agrees with the logged pass on 100.0 / 100.0 / 99.9%). Kaggle kept reporting it as
RUNNING for about seven hours after it had finished, so it was analysed on 2026-10-05 ~09:15. Analysis exactly as registered
(`scripts/passage_select.py`, committed before the outputs; 5-fold grouped out of fold; 80% intervals).
Report: `experiments/results/passage_select__Qwen3-VL-4B-Instruct__dev.{json,md}`.

| dev accuracy | E-VQA | InfoSeek | OK-VQA |
|---|---|---|---|
| never / always (top-5) | 26.7 / 52.9 | 17.3 / 25.7 | 59.4 / 38.6 |
| top-1 passage alone / `ev_keep` | 48.4 / 44.9 | 24.1 / 24.1 | 45.8 / 57.8 |
| **VGate-pick (benefit target)** | **55.7** | **27.3** | **60.2** |
| VGate-pick − always [80%] | +2.8 [+1.2, +4.4] | +1.6 [+0.4, +2.8] | +21.6 |
| VGate-pick − never [80%] | | | +0.8 [−0.1, +1.7] |
| correctness-target selector | 55.5 | 27.1 | 59.6 |
| VGate-pick without the verification pass | 48.0 | 26.8 | 60.0 |
| arg-max P(yes) (`verify`) / most confident / vote | 54.1 / 45.3 / 44.9 | 24.6 / 24.8 / 25.8 | 51.3 / 46.8 / 52.2 |
| oracle over the six candidates | 72.4 | 39.5 | 69.4 |
| tokens per query: always / VGate-pick | 1,075 / 4,933 | 1,019 / 4,567 | 985 / 4,583 |

- **Go / no-go: NO-GO.** The rule required VGate-pick − always ≥ +3.0 with the interval above zero on E-VQA **and** on
  InfoSeek. E-VQA reaches +2.8 and InfoSeek +1.6. (The OK-VQA condition, not more than 1.0 below never-retrieve, is met.)
  The method is not added to the paper as a positive result and no confirmatory run is planned.
- **Benefit vs correctness target at the passage level:** +0.2 [−1.0, +1.4], +0.2 [−0.9, +1.3], +0.6 [−0.1, +1.3] — no difference.
- **The verification pass carries the method.** Without it the selector is below always-retrieve on E-VQA (−4.9 [−6.8,
  −3.1]); with it the policy costs about 4.5 times the tokens of always-retrieve.
- **Headroom exists but is not reached:** the oracle over the six candidates is 19.5 / 13.8 / 30.8 points above
  always-retrieve; the selector recovers 2.8 / 1.6 of them on E-VQA / InfoSeek. Label-free rules (most confident answer,
  majority vote) fall below reading the top passage alone, as RMCD reports for its own baselines.
- **E4 (expectation recorded before the outputs) held:** the selector has little room on InfoSeek.

## Exploratory analyses of the pilot (v3 addendum E1–E3; `scripts/pilot_explore.py`, `experiments/results/pilot_explore.{json,md}`)
- **E1.** With the gold passage at ranks 2–5, the top-5 answer equals the answer from the gold passage alone in 47.4%
  (E-VQA, n = 228) / 47.5% (InfoSeek, n = 255) of cases and the answer from passage 1 alone in 36.8% / 39.2%.
- **E2 (reading-budget choice among closed-book, top-1 alone, top-5).** Allocator − always: +0.5 [−0.2, 1.2] (E-VQA, at
  1,174 vs 1,075 tokens), +1.2 [0.3, 2.1] (InfoSeek, at 1,019 vs 1,019 tokens), +20.2 on OK-VQA where it is 0.6 below
  never-retrieve. Reading the top passage alone loses 4.5 (E-VQA) and 1.6 (InfoSeek) points to top-5 at about two thirds
  of the tokens, and gains 7.2 on OK-VQA. Oracle over the three: 62.2 / 34.1 / 66.4. No usable gain.
- **E3 (early exit in retriever order).** −3.4 (E-VQA), +0.6 [−0.7, 1.8] (InfoSeek) against always-retrieve. No gain.
- Correction after the first run of the exploratory script: it charged always-retrieve the closed-book pass as well;
  tokens now follow the paper's ledger (always-retrieve pays the evidence pass only). No accuracy changed.

# Outcome log — PREREG v2d, selection with evidence-pass hidden states (J8; analysed 2026-10-05 ~09:20 IST)
Job `j8-evhidden` (build `2f8f009ff4ae`) ran 2.17 h; Kaggle reported it as RUNNING for many hours after it had finished.
Regenerated evidence-pass answers agree with the original passes on 100% of questions in all three datasets.
`scripts/extend.py --ev_hidden_run experiments/runs/J8` (J1 → J4 for E-VQA / InfoSeek, J6 → J6 for OK-VQA); held-out, 95% intervals.

| | accuracy | − always-retrieve | − VGate-select | keeps a_0 |
|---|---|---|---|---|
| E-VQA, Selector-H | 51.0 | −0.18 [−0.49, +0.08] | −0.47 [−0.91, −0.04] | 1.5% |
| E-VQA, Selector-HF | 51.0 | −0.18 [−0.77, +0.40] | −0.47 [−1.12, +0.12] | 10.8% |
| InfoSeek, Selector-H | 28.5 | +1.15 [+0.58, +1.72] | +0.58 [+0.03, +1.13] | 29.4% |
| InfoSeek, Selector-HF | 28.9 | +1.57 [+0.97, +2.25] | +0.99 [+0.42, +1.60] | 37.2% |
| OK-VQA, both | 59.6 | +19.65 [+18.13, +21.23] | 0.00 | 100% |

Reading: with both passes' hidden states (MMA-RAG's inputs; our linear reimplementation) selection gains 1.2–1.6 points on
InfoSeek, nothing on E-VQA, and reduces to never-retrieve on OK-VQA. Exploratory on E-VQA / InfoSeek; pre-specified
secondary on OK-VQA. No earlier value in `extend__*.json` changed (checked key by key; two NaN entries compare unequal to themselves).

# PREREG v4a — gates on a benchmark they were not fitted on (leave-one-benchmark-out; 2026-10-05, before any held-out number of it is computed)
**Why.** In mixed traffic the gate does about as well as a classifier that predicts the benchmark. If the gate had
learned something general about when retrieval helps, a gate fitted on two benchmarks should choose a sensible policy
on the third. A dev-only look (gates fitted on the dev sets of two benchmarks, applied to the dev set of the third)
says it does not: macro accuracy 35.6 (VGate-full), 39.6 (equal-feature error gate), 39.1 (always), 34.5 (never), 46.0
(better fixed policy per benchmark, which needs that benchmark's labels); Qwen2.5-VL-3B: 38.3, 40.7, 40.2, 33.9, 45.2.
On unseen OK-VQA every gate still reads 75–98% of queries. Development numbers; no claim.

**Analysis (`scripts/lodo.py`; no new model pass).** For each benchmark B: fit on the pooled dev sets of the other two
(no dataset identity), apply unchanged to B's held-out groups. Rules: never; always; the fixed policy that is better
on average over the two training benchmarks; "nearest regime" (predict which training benchmark the query resembles
from the closed-book hidden state and apply that benchmark's better fixed policy); error gates (P̂(wrong) > 0.5) with
VGate-post's and VGate-full's features; VGate-post, VGate-pre, VGate-full with the budget-free rule; oracle. Models:
Qwen3-VL-4B (J1 → J4, J6) and Qwen2.5-VL-3B (J2, J6). Reported per benchmark and as the macro mean, with 95%
image-cluster bootstrap intervals (stratified by benchmark) for the differences named below.

**Expectations stated in advance (reported as held or not held):**
- L1: no gate comes within 3.0 macro points of the better fixed policy per benchmark, for both models.
- L2: the equal-feature error gate is at least as accurate as VGate-full in macro accuracy (interval of
  error − benefit above −0.5), for both models. *This cuts against the paper's headline and is reported either way.*
- L3: on unseen OK-VQA every gate reads more than half of the queries, for both models.

## Outcome log — v4a (gates on an unseen benchmark; held-out, computed 2026-10-05 ~10:30 IST, `experiments/results/lodo__confirm.{json,md}`)
Registration and script committed first (`81f4f1a`). Gates fitted on the dev sets of two benchmarks, applied to the held-out
groups of the third; macro accuracy over the three held-out benchmarks.

| | Qwen3-VL-4B | Qwen2.5-VL-3B |
|---|---|---|
| never / always | 33.5 / 39.5 | 32.9 / 40.4 |
| better fixed policy per benchmark (needs that benchmark's labels) | 46.0 | 44.9 |
| error gate, VGate-full's features | 39.9 | 41.0 |
| VGate-full (benefit) | 36.0 | 38.8 |
| VGate-pre / VGate-post (benefit) | 34.8 / 34.9 | 38.0 / 36.3 |
| nearest training regime | 38.0 | 38.8 |
| error − benefit, full features [95%] | **+3.9 [+3.3, +4.5]** | **+2.2 [+1.6, +2.8]** |
| best gate − better fixed policy [95%] | −6.1 [−6.7, −5.5] | −3.9 [−4.4, −3.4] |

- **L1 held:** no gate comes within 3.0 macro points of the better fixed policy per benchmark.
- **L2 held:** under a change of benchmark the error target is the better one, by 3.9 and 2.2 points with equal features
  (3.7 and 3.9 without the hidden state). This is the reverse of the in-distribution result (+1.6 for the benefit target).
- **L3 held:** on unseen OK-VQA every gate reads 66–100% of the queries (accuracy 40–50% against 59.6 / 59.5 for never-retrieve).
- Per benchmark (Qwen3-VL-4B): benefit gates fitted on InfoSeek + OK-VQA read only 35–45% of E-VQA and lose 12–14 points to
  always-retrieve there; the error gate reads 90% and loses 0.8.
- **Reading.** A benefit label describes the model, the retriever *and the traffic it was collected on*; it does not carry over
  to a benchmark with a different help/harm balance, where the more generic error label degrades less. Neither gate recovers the
  right policy without labels from the target traffic. The paper's claim must therefore read: the benefit target is better
  *when the gate is fitted on the traffic it serves*; the one-sided-feedback result (v3b) is what makes that condition practical.

# PREREG v4c — the benchmark-identity control, the unseen-benchmark test and one-sided feedback on models 3 and 4 (2026-10-05, before jobs J11 / J12 are pushed)
**Why.** The mixed-traffic analysis (v2e) was specified after the OK-VQA outcome was known and exists for two models
(Qwen3-VL-4B, Qwen2.5-VL-3B). It carries the paper's central observation: a gate's gain in mixed traffic is matched by a
classifier that predicts the benchmark. A literature check (`docs/strengthening/R8_gates_vs_regime.md`) found no VLM
retrieval-gate paper that runs this control. This registration fixes the analysis and its expectations for two further
models **before any of their OK-VQA outputs exist**, which makes it a confirmatory replication.

**Data.** `kaggle/j11-okvqa-internvl` (InternVL3.5-4B-HF) and `kaggle/j12-okvqa-q3vl8b` (Qwen3-VL-8B-Instruct): the `ne`
(with hidden states) and `ev` passes on all 5,046 OK-VQA questions, same prompts, retrieval, splits and decoding as J6;
and the gold-alone (`oracle`) pass on all E-VQA and InfoSeek questions, as J4 ran it for the main model. Scoring as
before (OK-VQA: soft accuracy ≥ 0.6).

**Analyses (existing scripts, unchanged except for a model argument where one is missing).**
- `scripts/operating_point.py --model <m> --runs evqa=<J>:<J>,infoseek=<J>:<J>,okvqa=<J11|J12>:<same>`: one gate fitted on the
  pooled dev sets without dataset identity, budget-free rule, held-out groups; never, always, the per-benchmark fixed
  policy, the benchmark router, equal-feature error gate, VGate-pre / -post / -full, oracle.
- `scripts/lodo.py` (v4a) and `scripts/online_gate.py` (v3b) with the model added; `scripts/loss_decomposition.py` once the
  gold-alone files are placed next to the model's other runs.
- The OK-VQA endpoints of v2b (benefit − error target for the read gate and the hidden-state probe) are reported for both models.

**Expectations stated in advance, for each of the two models (held / not held is reported for each):**
- **C1** VGate-full − always-retrieve ≥ +3.0 macro points, 95% interval above 0.
- **C2** VGate-full does not exceed the per-benchmark fixed policy: the 95% interval of the difference lies below +1.0.
- **C3** The benchmark router is within 1.5 macro points of VGate-full in absolute value.
- **L1–L3** as in v4a (no gate within 3.0 macro points of the per-benchmark fixed policy on an unseen benchmark; the
  equal-feature error gate at least as accurate as VGate-full there; every gate reads more than half of unseen OK-VQA).
- **O1–O3** as in v3b (one-sided feedback: ≥ +3.0 over always-retrieve in every order over the last third; within 1.5 of
  the both-outcomes learner; within 2.0 of the dev-fitted gate).
- **D3** as in v3a (gold passage alone − all five ≥ 10 points when the gold section is at ranks 2–5 and untruncated).
If OK-VQA does not show the regime of the other two models (never-retrieve better than always-retrieve by at least 5
points), this is reported and C1–C3 are evaluated all the same. A crashed or partial job is not a negative result.

## Outcome log — v4c, model 3: InternVL3.5-4B-HF (J11; analysed 2026-10-05 ~13:30 IST). Model 4 (J12) is still running.
Job `j11-okvqa-internvl` (build `e33e36023c8d`, registered at `140be11` before the push) ran 2.15 h; all files complete, no error,
no NaN, no empty answer. OK-VQA held-out: never-retrieve 44.0, always-retrieve 26.3 (help 6.1%, harm 23.8%), so the regime
condition holds. Analyses with the existing scripts; `lodo.py` and `online_gate.py` only gained a model argument (`83ffb2e`,
before the outputs existed); `loss_decomposition.py` now evaluates D1 on the two models v3a registered it for.

| Expectation | Result | Held |
|---|---|---|
| C1 VGate-full − always ≥ +3.0, interval above 0 | +4.84 [+4.24, +5.42] (macro 38.0 vs 33.2) | yes |
| C2 VGate-full does not exceed the per-benchmark fixed policy (interval below +1.0) | −1.06 [−1.40, −0.70] (38.0 vs 39.1) | yes |
| C3 benchmark router within 1.5 of VGate-full | gate − router = −0.27 [−0.60, +0.06] (38.0 vs 38.3; router 89.7% accurate) | yes |
| L1 no gate within 3.0 of the per-benchmark fixed policy on an unseen benchmark | best gate −6.0 [−6.5, −5.4] | yes |
| L2 error gate at least as accurate as VGate-full there | error − benefit = +4.1 [+3.5, +4.6] (33.1 vs 29.1) | yes |
| L3 every gate reads more than half of unseen OK-VQA | 86–100% | yes |
| O1 one-sided feedback ≥ +3.0 over always in every order (last third) | 37.1 vs 31.4 | yes |
| O2 within 1.5 of the both-outcomes learner | 37.1 vs 37.2 | yes |
| O3 within 2.0 of the dev-fitted gate | 37.1 vs 37.2 | yes |
| D3 gold alone − all five ≥ 10 at ranks 2–5 (untruncated) | 26.5 [21.8, 31.5] | yes |

Also reported, as registered: VGate-full − equal-feature error gate in mixed traffic +1.86 [+1.40, +2.30]; OK-VQA benefit − error
target: read gate +0.25 [−0.03, +0.51] (not significant), hidden-state probe +0.56 [+0.06, +1.03]; the gate told whether the
gold item was retrieved against always-retrieve: −2.7 [−3.7, −1.8] (E-VQA), −0.5 [−1.1, 0.1] (InfoSeek); same passage text in both
prompts, gold alone − all five: 13.0 [10.6, 15.6].

# PREREG v5 — PILOT: passage selection with a released, trained critic (2026-10-05 ~23:30 IST, before job J13 is pushed)
**Status: pilot on development data; nothing here supports a confirmatory claim.**

**Why (evidence, not a guess).** The v3 pilot missed its bar because the reader's own zero-shot yes/no check picks the gold
passage only 71.2% of the time when it is among the five (retriever top-1: 60.5%), while about 88% is needed for a 5-point
gain (`docs/ISSUE_MATRICES.md`, issue 3). A trained verifier exists: `aimagelab/ReAG-Critic` (Compagnoni et al., arXiv
2511.22715; Qwen2.5-VL-3B fine-tuned on E-VQA / InfoSeek training passages of the ReflectiVA data), which its paper reports at
94.6% recall and 95.7% specificity for passage relevance on E-VQA (Table 6). Checked at the source on 2026-10-05. Caution
recorded in advance: in that paper's own pipeline (about 129 candidate passages per question, 5.7 kept) the critic adds only
+2.2 (E-VQA single-hop) and +0.1 (InfoSeek) to a non-fine-tuned 3B reader (Table 4), and our candidates are PreFLMR's top-5,
i.e. hard negatives, on which its specificity is unknown. The critic is a **third-party supervised component trained on
these benchmarks' training data**; any result with it is reported as such, not as training-free.

**Data.** Job `kaggle/j13-critic-pilot`: the critic scores each of the top-5 retrieved passages of the three dev splits
(condition `cv_k`: its model card's system and user prompts verbatim, passage as `# <page title>:\n<text>`, 300-word cap;
P(Yes) at the next token, normalised over yes / no tokens and also stored raw). No reader pass is needed: the
single-passage answers of the main model already exist (J10).

**Analysis 1 (registered script, unchanged).** `scripts/passage_select.py` on a run folder `J13` that holds J10's reader
outputs with the verification files (`pv_k`) replaced by the critic's. Same selector, features, baselines and
**the same go / no-go rule as v3**: VGate-pick − always-retrieve ≥ +3.0 points on E-VQA **and** InfoSeek with the 80%
interval above 0, and not more than 1.0 below never-retrieve on OK-VQA.

**Analysis 2 (`scripts/critic_pilot.py`, written before the outputs).**
- how often the critic's arg-max passage is the gold one when the gold passage is among the five (E-VQA, InfoSeek);
- **critic-pick**, a rule with no fitted parameter: answer from the passage with the highest critic score if its raw
  P(Yes) exceeds 0.1 (the card's threshold), else the closed-book answer;
- the benchmark-identity control for it: macro accuracy over the three dev sets against the better fixed policy per
  benchmark, and the same for a selector fitted on the pooled dev sets without dataset identity (out of fold).
Expectations stated in advance: **P1** arg-max critic picks the gold passage in at least 80% of gold-in-top-5 cases on
E-VQA; **P2** critic-pick beats always-retrieve by at least 3.0 points on E-VQA (80% interval above 0); **P3** critic-pick's
macro accuracy over the three benchmarks exceeds the per-benchmark fixed policy by at least 2.0 points, which no gate has done.

**If GO or P3 holds:** a confirmatory registration (v5c) before any held-out critic output exists: critic scores on the
held-out splits (about 4 GPU-h), single-passage reads of the selected passages, a second reader model, and the claim "the
audit's prescription — select evidence rather than gate — is confirmed by intervention". **If not:** logged as a negative
result; the paper keeps its present claim (v).

## Outcome log — v4c, model 4: Qwen3-VL-8B-Instruct (J12; analysed 2026-10-05 ~23:40 IST)
Job `j12-okvqa-q3vl8b` (build `e33e36023c8d`) ran 4.47 h; all three parts done, no error, no NaN, no empty answer. OK-VQA
held-out: never-retrieve 61.9, always-retrieve 44.3 (help 4.0%, harm 21.7%). Same scripts and arguments as for model 3.

| Expectation | Result | Held |
|---|---|---|
| C1 VGate-full − always ≥ +3.0, interval above 0 | +5.09 [+4.55, +5.58] (macro 48.7 vs 43.6) | yes |
| C2 VGate-full does not exceed the per-benchmark fixed policy (interval below +1.0) | −0.80 [−1.09, −0.52] (48.7 vs 49.5) | yes |
| C3 benchmark router within 1.5 of VGate-full | gate − router = −0.49 [−0.76, −0.22] (48.7 vs 49.1; router 92.2% accurate) | yes |
| L1 no gate within 3.0 of the per-benchmark fixed policy on an unseen benchmark | best gate −5.8 [−6.4, −5.3] | yes |
| L2 error gate at least as accurate as VGate-full there | error − benefit = +3.6 [+3.1, +4.1] (43.6 vs 40.0) | yes |
| L3 every gate reads more than half of unseen OK-VQA | 76–100% | yes |
| O1 one-sided feedback ≥ +3.0 over always in every order (last third) | 47.8 vs 42.9 | yes |
| O2 within 1.5 of the both-outcomes learner | 47.8 vs 48.2 | yes |
| O3 within 2.0 of the dev-fitted gate | 47.8 vs 48.6 | yes |
| D3 gold alone − all five ≥ 10 at ranks 2–5 (untruncated) | 24.2 [19.6, 28.9] | yes |

Also reported: VGate-full − equal-feature error gate in mixed traffic +1.34 [+0.93, +1.73]; OK-VQA benefit − error target: read
gate +1.20 [+0.76, +1.63], hidden-state probe +1.53 [+1.06, +2.00]; gate told whether the gold item was retrieved against
always-retrieve −1.0 [−2.0, 0.0] (E-VQA), 0.0 [−0.7, 0.6] (InfoSeek); same passage text in both prompts, gold alone − all five 12.8 [10.7, 15.1].

**v4c summary (both models).** All twenty registered expectations held. Across the four models: VGate-full − always-retrieve
+3.7 to +5.7; the benchmark router is 0.3–0.5 points ahead of the gate and the per-benchmark fixed policy 0.8–1.1 ahead; on an
unseen benchmark the error target beats the benefit target by 2.2–4.1 points and the best gate is 3.9–6.1 below the better fixed
policy; the one-sided-feedback learner ends within 0.8 points of the fitted gate. On OK-VQA the read-gate benefit − error effect
is significant on two of the four models (Qwen2.5-VL-3B, Qwen3-VL-8B) and the hidden-state probe's on all four.

# Outcome log — PREREG v5 pilot: passage selection with the released critic (J13; analysed 2026-10-06 ~00:50 IST)
Job `j13-critic-pilot` (build `5bb218ce33f3`, registered at `7cb56f6` before the push) ran 1.44 h; the 16-question smoke run
passed and all 15,000 critic scores were written. Run folder `experiments/runs/J13` = J10's reader outputs with the `pv_k`
files replaced by the critic's (originals under `critic_raw/`). Development data, 80% intervals. Reports:
`experiments/results/passage_select__Qwen3-VL-4B-Instruct__dev__critic.{json,md}`, `experiments/results/critic_pilot.{json,md}`.

| | E-VQA | InfoSeek | OK-VQA |
|---|---|---|---|
| always / never | 52.9 / 26.7 | 25.7 / 17.3 | 38.6 / 59.4 |
| **Analysis 1** VGate-pick with the critic as the verifier − always | **+1.8 [+0.2, +3.3]** | **+1.4 [+0.3, +2.5]** | +21.5 (+0.7 over never) |
| (v3, the reader's own yes/no check as the verifier) | +2.8 [+1.2, +4.4] | +1.6 [+0.4, +2.8] | +21.6 |
| arg-max critic score ≥ 0.5, else closed-book (`verify`) − always | +3.4 [+1.5, +5.2] | −3.5 [−5.2, −1.8] | −4.9 against never |
| **Analysis 2** critic-pick (card's threshold 0.1) − always | +2.9 [+1.1, +4.7] | −1.8 [−3.5, −0.1] | −10.3 against never |
| arg-max critic is the gold passage, given gold in top-5 | 71.8% | 52.6% | |
| critic at its threshold on our top-5: recall / specificity | 95.5% / 41.4% | 30.5% / 93.7% | |
| AUROC for the gold passage (reader's own check: 0.815 / 0.749) | 0.878 | 0.780 | |

- **Go / no-go (same rule as v3): NO-GO.** +1.8 and +1.4 against a bar of +3.0, at about 6.3 times the tokens of always-retrieve.
- **P1 not held:** the critic's arg-max passage is the gold one in 71.8% of gold-in-top-5 cases on E-VQA (bar 80%); the
  reader's own zero-shot check gets 71.2% and the retriever's first rank 60.5%.
- **P2 not held:** critic-pick − always = +2.9 [+1.1, +4.7] on E-VQA (bar +3.0).
- **P3 not held:** macro accuracy of critic-pick 42.9 against 46.0 for the better fixed policy per benchmark (−3.1 [−4.0, −2.1]);
  one benefit-trained selector without dataset identity 45.6 (−0.4 [−1.2, +0.5]). Nothing beats the benchmark-identity control.
- **Why.** The caution recorded in the registration applies: on PreFLMR's five candidates, which are hard negatives, the
  critic keeps its recall (95.5%) but its specificity falls to 41.4% (95.7% on the candidate pool of its own paper; it accepts
  3.1 of 5 passages per E-VQA question). On InfoSeek it rejects most of our chunks, including those of the right page (recall
  30.5%), and on OK-VQA it still accepts a passage for 38.5% of questions.
- **Decision (as registered):** negative result; no held-out run; the paper keeps claim (v) and may state that a critic trained
  on these benchmarks does not identify the right passage among the retriever's top five more often than the reader's own check.

## Notes added after claim check 7 (2026-10-06 ~01:10 IST; `docs/experiments/reviewer_simulation_7.md`)
- **Multiplicity.** `scripts/multiplicity.py` now includes the four OK-VQA endpoint tests of models 3 and 4 that v4c reports
  (benefit − error for the read gate and for the hidden-state probe). Family: 22 tests; 17 intervals exclude zero; **16 remain
  significant after Holm correction**. The one lost is InternVL3.5-4B's OK-VQA probe effect (+0.56 [0.06, 1.03], p = 0.023 against
  a threshold of 0.0083). The expectations C1–C3, L1–L3, O1–O3, D3 are threshold checks, not tests in this family.
- **An exception not in the v4c outcome log.** On OK-VQA the answer selector after reading (VGate-select) beats never-retrieve for
  InternVL3.5-4B by +1.38 [+0.87, +1.88] (45.4 vs 44.0); for the other three models it only recovers never-retrieve (0.0, +0.15
  [−0.12, 0.42], −0.02 [−0.35, 0.32]). It is the only policy in the repository that beats a better fixed policy by more than a
  point with the interval above zero. It reads evidence for every query. The paper states it.
- **"Preregistered" for models 3–4** means: registered before their OK-VQA outputs existed; their E-VQA and InfoSeek held-out
  outputs had been scored on 2026-10-03/04. The paper says so.
- Wording changed in the paper accordingly (per-benchmark statement marked exploratory; ranges across models; the error gate ends
  within 0.7 points of always-retrieve on an unseen benchmark; no passage selector recovers 3 points on both benchmarks, with the
  critic's arg-max rule at +3.4 on E-VQA and −3.5 on InfoSeek as the logged exception). The replication table was moved into the text.

# PREREG v6 — PILOT: identifying the pictured entity by matching against the candidates' page images (2026-10-06, before any image is embedded)
**Status: pilot on development data (E-VQA dev), no model pass; nothing here supports a confirmatory claim.**

**Why (evidence).** Both selection pilots failed because the right passage is identified only ~72% of the time among the
retriever's five, with text-only candidates (v3: the reader's own check; v5: a trained critic). Published results say the
missing signal is the candidates' *images*: a trained reranker gains nothing from the query image when candidates are
text-only but about ten points of recall@1 when candidate images are added (OMGM Table 6, as collected in
`docs/strengthening/R10_entity_verification_rerank.md`); image-to-image retrieval is EchoSight's first stage; Google Lens
alone reaches recall@1 47.4 on E-VQA. Our candidates are text sections of Wikipedia pages; each page has a lead image.

**Data.** For the unique page titles among PreFLMR's top-20 passages of the 1,000 E-VQA dev questions (plus gold pages, for
analysis only), the lead thumbnail (336 px) from the public Wikipedia page-image API; kept outside the repository and not
redistributed. Frozen encoders: DINOv2-base and CLIP ViT-B/16; score = cosine similarity between the question's image and
a candidate page's lead image (pages without an image get the question's mean similarity).

**Analysis (`scripts/kbimage_pilot.py`, written before any embedding is computed).**
- Page level: how often the gold page is ranked first among the top-20 candidates' pages (when it is among them) by the
  retriever, by image similarity, and by the sum of their within-question z-scores (no fitted parameter).
- Passage level: among the top-5 passages, how often the passage chosen by the fused score (ties within a page broken by
  retriever rank) is the gold one when the gold passage is among the five; for comparison 60.5% (retriever), 71.2%
  (reader's check), 71.8% (trained critic).
- Answer level: accuracy of reading the chosen passage alone (single-passage answers from J10) against always-retrieve,
  80% image-cluster bootstrap; and the same with the choice widened to the best-scored page among the top-20 where its
  best-ranked passage is among the five already read (no new reader pass in this pilot).
**Go / no-go (fixed now):** GO if, for the better of the two encoders chosen on these data, the fused rule picks the gold
passage in at least 80% of gold-in-top-5 cases **and** reading its choice alone beats always-retrieve by at least 3.0 points
with the 80% interval above 0. A GO leads to a registration on InfoSeek and the held-out splits (which need reader passes
for passages outside the top-5 and images for the held-out candidates). Otherwise it is logged as negative.

**Amendment to v6 (2026-10-06, before any image is embedded).** E-VQA's landmark photographs come from the Google Landmarks
collection, which is drawn from Wikimedia Commons, so a question's photograph may be the same file as (or a near copy of)
a candidate page's lead image; species photographs come from iNaturalist and cannot be. The script therefore also reports,
per image source, the share of questions whose best-matching candidate image has cosine similarity above 0.95 and the gold
pick rate and gain. **A GO additionally requires the gain on the iNaturalist questions to be positive**; a gain that comes
only from near-copies on the landmark questions is logged as a property of the benchmark, not as a method. The Wikimedia
API allows ten requests a minute to clients that send no contact details, so the page-image lookups run at that rate.

# PREREG v7 — a fourth benchmark where retrieval both helps and hurts: CVQA (2026-10-06, before any CVQA output exists)
**Why (evidence).** On E-VQA, InfoSeek and OK-VQA one outcome of retrieval is rare, so a read/skip gate has at most 3.5–6.1
points of headroom over the better fixed policy and no gate reaches one point of it (claim (ii)). Claim check 7 objected
that benchmark, retriever and question style change together across our three benchmarks. The text literature says where
a gate should pay: "closed-book accuracy is nontrivial, retrieval has enough headroom to rescue low-confidence queries, and
retrieval can also harm high-confidence queries" (Know Before You Fetch, arXiv 2606.29959); a gate beats both fixed policies
when always-retrieve is at or below never-retrieve (TARG, arXiv 2511.09803: +1.4 to +3.0 EM). M4-RAG (arXiv 2512.05959,
Table 3) reports exactly that regime for small VLMs on CVQA: closed-book 56–59, with retrieved text 51–65 depending on the
retriever, with oracle context 94–95 (`docs/strengthening/R9_benefit_signals_and_settings.md`, part 2). The numbers there are
for other readers (a Thinking 4B model among them); the help and harm rates of our reader are unknown until measured.

**Data.** CVQA (HF `afaji/cvqa`, 10,374 four-option questions; English question and options; labels public). Images are
read inside the GPU job and not stored. Evidence: the M4-RAG Wikipedia corpus (HF `davidanugraha/M4-RAG`, config `cvqa`,
306,794 pages in many languages). Retrieval (`src/vgate/cvqa.py`): dense bge-m3; query = the reader's entity name for the
image (the existing `ent` prompt) + the question, without the options; a page is embedded as its title and first 150 words;
top-5 pages, the best 120-word chunk of each (first 20 chunks; for scripts written without spaces two characters count
as a word; a passage is capped at 220 retriever tokens) as evidence; top-20 page scores as the retriever-score
features. Split: questions grouped by image (hash of the image bytes), seed 2027, development = the first groups reaching
2,000 questions, held-out = the rest (`data.grouped_split`).
**Reader and prompts.** Qwen3-VL-4B-Instruct, fp16; `NE_MC` / `EV_MC` in `src/vgate/prompts.py` (id `PROMPT_ID_V7`); an
answer is correct when its option letter matches (`evaluation.mc_letter`). Gate features and gates are those of the paper,
unchanged (free token statistics, retriever-score features, PCA-64 of the closed-book hidden state; error label vs benefit
label; read iff P(help) > P(harm) or P(wrong) > 0.5).
**Jobs.** J14: entity pass and retrieval for all questions, closed-book and evidence passes on the development split.
J15: the two passes on the held-out split. **J15 is run only if R1 holds**; otherwise CVQA is logged from the development
split as a fourth benchmark without two-sided outcomes and J15 is NOT_RUN (its tests presuppose R1).

**Analysis (`scripts/cvqa_audit.py`, committed with this text and dry-run on synthetic data only).** Gates fitted on the
development split, frozen, applied once to the held-out split; 95% image-cluster bootstrap. Policies: never, always, the
better fixed policy (chosen on development data), told the category (10) / told the country-language subset (39) and the
same with the segment predicted from the gate's own features (the within-benchmark form of the benchmark-identity control:
retriever and question style are constant here), error gates, VGate-post / pre / full, oracle.
- **R1 (regime, development split):** retrieval helps on at least 8% of questions and hurts on at least 8%.
- **G1:** VGate-full beats the better fixed policy by at least 1.0 point, with the interval above 0 — the bar no gate met
  on the three other benchmarks.
- **G2:** VGate-full beats the error gate with the same features (interval above 0); the nAUC difference is reported too.
- **G3:** VGate-full beats the told-the-category policy (interval above 0), i.e. its gain is not the category's.
These three are a separate family (Holm over G1–G3); they are not added to the 22 tests of the main family. Whatever the
outcome, the paper reports it: if G1 holds, gating pays where the audit's diagnostics (both outcomes common) say it can; if
R1 holds and G1 does not, headroom exists and the gate's signals do not reach it. One reader and one retriever; a second
reader is run only if time allows and would be labelled a replication.

# PREREG v8 — published gates through the audit (2026-10-06, before these gates are run on our questions)
**Why (evidence).** Claim check 7's first objection: the audit covers only our own gates. Accepted audit papers run other
people's published methods (`docs/strengthening/R12_accepted_patterns.md`, part C: 8 pipelines + 27 estimators; 10 routers;
named papers reproduced). `docs/strengthening/R13_released_gates.md` lists which published VLM retrieval gates can be run as
released. Their own papers already hint at the pattern our control tests: ReflectiVA's token fires on 88.4% / 100% of
E-VQA / InfoSeek and 0% of GQA; Chen et al.'s prompt-based gate retrieves on 0–31% and their trained gate on 35–99% by
dataset; UniversalRAG's router labels are the benchmark.

**Gates (nothing fitted by us; each read at its published threshold).**
- `chen` — the search-need prompt of Chen et al. (arXiv 2502.18023, Appendix A.1; "true" / "false"), run with our main
  reader; text copied from the paper's HTML, question then image as printed. The paper does not say whether its
  "Prompt-based" baseline used this text unchanged; we call it "the search-need prompt of Chen et al.", not their baseline.
- `urag` — UniversalRAG's training-free router prompt (`route/prompt.py`, copied verbatim), text query only as published,
  run with our reader instead of GPT-5; "no" = skip, any modality = retrieve.
  Both: `src/vgate/prompts.py` (`PROMPT_ID_V8`), decision = P(retrieve words) > P(skip words) at the first word of a short
  greedy reply (`models.first_token`); the reply text is kept and its agreement with the rule is reported. Job J16: every
  question of the three benchmarks (13,433), Qwen3-VL-4B-Instruct.
- `reag_critic` — the released ReAG-Critic checkpoint as a use-the-evidence gate after retrieval: read iff any of the five
  passages has P("Yes") > 0.1 (the card's threshold; the question-level rule is ours). Its scores exist from J13 on the
  **development split only** and its passage-selection results are known (v5), so this row is exploratory.
- `reflectiva` — the released ReflectiVA checkpoint's `[Retrieval]` / `[No Retrieval]` first token (J17, own kernel, after
  the weekly quota resets on 2026-10-10); registered here so that the analysis is fixed before it runs. If J17 cannot be
  made to run it is reported as BLOCKED, not as a result.

**Analysis (`scripts/external_gates.py`, committed with this text).** A gate's decisions are applied to the main reader's
closed-book and top-5 answers on the held-out split (95% image-cluster bootstrap, stratified by benchmark in mixed traffic):
read rate and accuracy per benchmark against never, always and the better fixed policy (chosen on development data); in
mixed traffic against the told-the-benchmark policy; and the gate against **its own majority decision on each benchmark**
(every decision replaced by what the gate does most often on that benchmark, majority taken on the development questions).
Also reported: the AUROC of the gate's score for help vs harm and for a wrong closed-book answer, within each benchmark.
**Expectations, for each gate:**
- **X1:** on no benchmark does following the gate beat the better fixed policy by 1.0 point or more.
- **X2:** in mixed traffic the gate does not beat the told-the-benchmark policy (upper end of the interval below +1.0).
- **X3:** replacing the gate's decisions by its per-benchmark majority changes mixed-traffic accuracy by less than 1.0 point.
A gate for which X1–X3 hold is, on these benchmarks, indistinguishable from a rule that only knows the benchmark; a gate
for which they fail carries per-query value that our own gates did not find, and the paper will say so. Caveats to state:
a gate tuned or prompted for web search is applied here to knowledge-base retrieval; `urag` is run with a 4B reader, not
GPT-5; decisions are applied to our reader's answers, not to the answers of each gate's own system.

**Amendment to v8 (2026-10-06, before J16 is pushed; no output of these gates exists).** A dry run on synthetic gate
outputs (random decisions around a benchmark-level rate) showed that X3 as written fails for a gate with no per-query value,
because replacing noisy decisions by the benchmark majority *raises* accuracy. X3 is therefore one-sided: **the gate does
not beat its own per-benchmark majority decision by 1.0 point or more.** The signed difference is reported; a negative value
means the gate's within-benchmark variation costs accuracy.

**J14 status: FAILED (infrastructure), 2026-10-06 11:04.** The job stopped in its retrieval step after 61 minutes with
`CUDA driver error: operation not permitted` in one of the two threads that embedded the corpus; the entity pass had finished
(10,374 names) and no closed-book or evidence answer existed. Not a result. Rerun as **J14b** with the embedding in one worker
process per GPU (as the reader passes run), an embedding self-test before anything long, and the exact search on the CPU;
J14's entity names, records and frozen splits (development 2,001 questions, held-out 8,373; 4,752 images) are reused. Nothing
else in v7 changes. J15 will read J14b's records.

**v7 outcome on the development split (J14b, 2026-10-06 14:50; `experiments/results/cvqa_audit__dev.md`; n = 2,001, gates out of
fold, 80% intervals; development data, no confirmatory claim).** Closed-book 66.3, always-retrieve 60.7; retrieval helps on
8.55% of questions and hurts on 14.09%; oracle gate 8.55 above the better fixed policy. **R1 held → J15 is run.** Development
estimates of the held-out tests: VGate-full − better fixed policy +1.20 [+0.20, +2.30]; − error gate with the same features
+2.10 [+1.31, +2.93]; − told the category +2.35 [+1.38, +3.35]; − told the subset +1.05 [+0.25, +1.91]. Retrieval: 87% of
top-1 pages are English. With evidence the reader's reply names no option on 7.5% of questions ("None of the provided
options …"), which is 24.5% of the harm cases; by the registered rule such a reply is wrong.

**Amendment to v7 (2026-10-06, after the development outcome, before J15 is pushed).** A registered sensitivity analysis is
added because a quarter of the harm on development data is the reader declining to choose: `scripts/cvqa_audit.py --fallback`
repeats every comparison with an evidence reply that names no option replaced by the closed-book answer (a rule any system
could apply without a gate). On development data: always-retrieve 64.2, harm 10.6%, VGate-full − better fixed policy +1.95
[+0.90, +3.08], − error gate +0.90 [+0.20, +1.63], − told the category +0.65 [−0.15, +1.46]. G1–G3 remain the registered
tests, on the registered labels; the fallback versions (G1f–G3f, same thresholds) are reported next to them, and the paper
states both. No other change.

**v8 outcome (J16, 2026-10-06; `experiments/results/external_gates__{confirm,dev}.md`; main reader, held-out split unless noted,
95% intervals).** Both prompts ran as registered: the reply opened with one of the expected words on every question (probability
mass 1.00), and the reply's words agree with the probability rule on at least 99.6%.
- **Search-need prompt of Chen et al.** Reads 23.0% / 48.3% / 40.7% of E-VQA / InfoSeek / OK-VQA. Against the better fixed
  policy: −23.1, −3.2, −7.4 (**X1 held**). Mixed traffic 34.8: −11.2 [−12.2, −10.2] against told-the-benchmark (**X2 held**),
  −4.7 [−5.7, −3.6] against always-retrieve. Against its own majority decision per benchmark (skip on all three, i.e.
  never-retrieve) +1.34 [+0.74, +1.99]: **X3 NOT held.** Reading (ours): the majority comparison is not the right null for a gate
  that reads a middling share of a benchmark on which retrieval helps on average. Added after the results (exploratory): against
  random decisions at its own read rate on each benchmark the gate is +0.29 [−0.20, +0.84] (−1.5 on E-VQA, +1.7 on InfoSeek,
  +0.6 on OK-VQA). AUROC of its score for help vs harm: 0.43 / 0.63 / 0.50.
- **UniversalRAG router prompt** (run with our 4B reader, not GPT-5). Reads 99.5% / 98.8% / 96.2%: it is always-retrieve
  (+0.09 [−0.03, +0.20]); −6.5 [−7.0, −5.9] against told-the-benchmark. **X1, X2, X3 held.**
- **ReAG-Critic as a use-the-evidence gate, development split (exploratory).** Reads 95.1% / 37.2% / 38.5%; +0.2, −2.9, −9.5
  against the better fixed policy; mixed 41.9: −4.1 [−5.1, −3.0] against told-the-benchmark, +2.9 against always-retrieve,
  +0.8 [+0.0, +1.5] against random decisions at its read rates. Its score separates help from harm on the two benchmarks it
  was trained on (AUROC 0.72 / 0.68) and not on OK-VQA (0.44). **X1, X2, X3 held.**
- ReflectiVA (J17): smoke run pushed 2026-10-06 14:45; the full run waits for the quota reset.
Eight of the nine expectations held. On these benchmarks no released gate comes within four points of the told-the-benchmark
policy, and none beats a benchmark's better fixed policy by one point.

**Deviation from v6, logged before any image is embedded (2026-10-06 15:00).** Wikimedia rate-limits clients that send no
contact details; the image download ran for its full two-hour allowance and stopped with every page of the top-5 passages
(2,951 with a lead image) and every gold page (475) on disk, but only half of the pages of ranks 6–20 (3,878 of 7,763). The
pilot is therefore run with **five candidates per question instead of twenty** (`scripts/kbimage_pilot.py --n 5`): the
passage-level and answer-level analyses and the go / no-go rule are as registered (they use the top-5 only); the page-level
ranking among twenty candidates is NOT_RUN.

**J17 smoke run (2026-10-06 14:45–14:58, 16 questions per benchmark; a feasibility check, not a result).** The released
ReflectiVA checkpoint loads and runs on two T4s in fp16 at 0.8–0.9 s per question. Its first generated token was one of
`[Retrieval]` / `[No Retrieval]` on all 48 questions (probability mass 0.97 on the two). **Deviation to state with the result:**
the release pins transformers 4.43.1, whose tokenizers wheel does not build on the Kaggle image (Python 3.13); the checkpoint
was run with the image's transformers 5.16.1 through the release's own loader and prompt code, and the token is read greedily
(the release samples at temperature 0.2). The full run (13,433 questions, about 3.2 GPU-hours) waits for the quota reset.
**Second deviation from v6 (2026-10-06 15:02, before any number was printed or saved).** The first run of the pilot stopped
with a loading error at the CLIP encoder (the cached LAION ViT-B/16 file is in open_clip format); it wrote nothing. The CLIP
ViT-B/16 encoder is therefore OpenAI's (`openai/clip-vit-base-patch16`); DINOv2-base is unchanged.

**v6 outcome (2026-10-06 15:15; `experiments/results/kbimage_pilot.md`; E-VQA development split, n = 1,000, five candidates
per question, 80% intervals; pilot, no confirmatory claim): NO-GO.**
- Page level, when the gold page is among the five candidates' pages (60.5% of questions; 98.3% of those gold pages have a
  lead image): the gold page is ranked first by the retriever in 64.5% of cases, by image similarity in 75.2% (DINOv2-base) /
  66.0% (CLIP ViT-B/16), by the fused score in 74.5% / 70.9%.
- Passage level: the fused pick is the gold passage in **70.0%** (DINOv2) / 66.0% (CLIP) of cases, against 60.5% for the
  retriever's first rank, 71.2% for the reader's own check (v3) and 71.8% for the trained critic (v5). Bar: 80%.
- Answer level: reading the fused pick alone gives 51.7 (DINOv2) against 52.9 for always-retrieve: −1.2 [−2.9, +0.4]. Bar: +3.0.
- By image source (DINOv2): iNaturalist 71.5% gold picks (retriever 62.2%), gain 0.0; landmarks 68.5% (58.7%), −2.9; a
  near-copy of a candidate's lead image occurs for 0.0% / 2.7% of questions, so the benchmark artefact feared in the amendment
  is small with present-day Wikipedia lead images.
Reading: a third, independent verifier (the reader's check, a trained critic, and now image matching against the candidate
pages) stops at about 70–75% where about 88% is needed. Matching against one lead image per page identifies the page more
often than the retriever does, but not the section, and not often enough. The light fused reranker that was conditional on a
GO is not built. Page-level ranking among twenty candidates: NOT_RUN (images incomplete, see the deviation above).

**Correction and exploratory diagnosis of the v6 outcome (2026-10-06 15:40; `scripts/kbimage_diag.py`, DINOv2-base, same
development questions; nothing here is a registered result, and the sweeps are in-sample).** The "Reading" above said that
image matching finds the page but not the section. That is wrong: the gold page picked with the wrong section is 4.7% of
questions. What the data show:
- The tested rule reads **one** passage alone. Where it picks the gold passage (40.4% of questions) it gains 11.8 points over
  reading all five (91.3 vs 79.5); where the gold page is among the five and another page is picked (15.4%) it loses 31.8
  (26.0 vs 57.8); where the gold page is not among the five (39.5%) nothing changes (22.0 vs 23.0). A wrong pick costs almost
  three times what a right pick gains, so the rule breaks even at about 73% correct picks and is at 70%.
- Ceiling of any re-pick among the five: an oracle passage read alone gives 62.0 against 52.9 (+9.1).
- In-sample: a fusion weight of 2 on the image score gives 74.9% gold picks and 53.8; using the pick only when the image
  match is confident (48% of questions), else all five, gives 53.9–54.0. Both are optimistic and below the bar of +3.
- **Not tested, and not covered by the NO-GO:** (a) re-ranking the top-20 so that the gold page enters the five that are read:
  the gold page is at ranks 6–20 for 18.2% of questions (PreFLMR page recall 60.5% / 70.2% / 78.7% at 5 / 10 / 20), all-five
  accuracy is 72.4 when the gold page is read and 23.0 when it is not, so ten such questions moved in would be worth about
  4.9 points; only 55% of the rank 6–20 candidates have an image on disk, and accuracy needs a new reader pass. (b) Reordering:
  the picked passage first, all five kept; reading all five is right 84.2% of the time with the gold passage first and 58.3% at
  ranks 2–5, the fused pick would move it to rank 1 for 7.5% of questions and away for 2.0%, about +1.4 points if rank alone
  drove that gap; needs a reader pass. (c) Several images per page, other encoders, a fitted fusion.
The NO-GO therefore covers "pick one of the five by image match and read it alone", not candidate images as such.

**v7 outcome on the held-out split (J15, 2026-10-06 16:40; `experiments/results/cvqa_audit__confirm.md` and
`…__confirm__fallback.md`; n = 8,373, gates fitted on the 2,001 development questions and applied once; 95% intervals).**
Regime as on development data: closed-book 65.6, always-retrieve 59.9; retrieval helps on 8.3% of questions and hurts on 14.0%;
an oracle gate is 8.3 points above the better fixed policy (never-retrieve).
- **G1 NOT held.** VGate-full 65.96: +0.38 [−0.20, +0.96] over the better fixed policy (bar: +1.0 with the interval above 0).
  VGate-post +0.60, VGate-pre +0.27 (not registered tests).
- **G2 held.** VGate-full − error gate with the same features +0.99 [+0.34, +1.63]; nAUC +1.70 [+1.27, +2.10].
- **G3 NOT held.** VGate-full − told the category +0.38 [−0.21, +0.94] (the told-the-category policy reads 6% of questions and
  equals never-retrieve to two decimals); − told the subset +1.16 [+0.55, +1.75].
- Registered sensitivity analysis (a reply that names no option, 7.6% of questions and 22.4% of the harm cases, falls back to
  the closed-book answer; always-retrieve then 63.1, harm 10.8%): **G1f NOT held** (+0.86 [+0.23, +1.48]: above zero, below
  one point), **G2f NOT held** (+0.01 [−0.57, +0.57]), **G3f held** (+1.76 [+1.10, +2.40]).
One of three registered tests held (Holm over G1–G3: G2 remains significant). The development estimate of G1 (+1.20) did not
survive. Reading: on a benchmark with one retriever and one question style, where both outcomes are common and an oracle gate
has 8.3 points of headroom, the gate still does not lead the better fixed policy by one point; the within-benchmark result of
the other three benchmarks is not an artefact of their one-sided regimes. The benefit label again ranks queries better than
the error label with identical features, on the registered labels only.

**Diagnosis of the v7 outcome (exploratory; `scripts/cvqa_diag.py`, `experiments/results/cvqa_diag.md`; 2026-10-06 17:10).**
Why G1 failed, from the data:
- The gate reads 19.8% of held-out questions. Among them retrieval helps on 16.6% (base rate 8.3%) and harms on 14.7% (base rate
  14.0%): it finds help but does not avoid harm. Net +0.38.
- The limit is the ranking, not the threshold: AUROC of the predicted benefit for help vs harm 0.626; the best threshold chosen
  with the held-out labels gives +0.73 at a 9% read rate.
- More labelled traffic helps and saturates below the bar: fitted on 500 / 1,000 / 2,001 development questions −0.33 / −0.04 /
  +0.38; fitted on about 6,700 held-out questions and scored on the rest, +0.73 [+0.21, +1.24].
- The hidden-state components do not help at this sample size (VGate-post +0.60, VGate-full +0.38); error gates are negative.
Candidate fixes, with my estimate of the chance that each leads the better fixed policy by one point on held-out data:
(a) a pre-read gate with more labels (10,000+): about 25% (the curve above flattens at +0.7); (b) other pre-read signals:
below 15% (harm is not visible before reading: the harm rate among the gate's reads equals the base rate); (c) **deciding
after reading**: on development data, keeping the answer with the higher first-token probability gains +2.50 [+1.42, +3.69]
(80% interval) over never-retrieve with no fitted parameter, a fitted selector +1.70; about 85%, because the rule has no
parameter to overfit and its development interval is well above one point. It costs both passes on every question.
The same rule on the three open-ended benchmarks' development splits (four models) is at or below the better fixed policy
(main model −1.9 / +0.3 / −10.6 on E-VQA / InfoSeek / OK-VQA): with free-form answers the first-token probability is not a
comparable confidence across the two passes. So (c) is a fix for closed-set answers only.

# PREREG v7c — CVQA: choosing the answer after reading (2026-10-06 17:15, before the rule is applied to any held-out answer)
**Rule (no fitted parameter, fixed on development data above):** read the evidence for every question; answer with the evidence
reply iff it names an option and its first-token probability exceeds that of the closed-book reply; otherwise answer
closed-book. Script: `scripts/cvqa_select.py`, committed with this text. Held-out split of v7 (n = 8,373), 95% image-cluster
bootstrap; the held-out closed-book and evidence answers exist (J15) and their accuracies are known from v7; the two replies'
probabilities have not been compared.
- **S1:** the rule beats the better fixed policy (never-retrieve) by at least 1.0 point, with the interval above 0.
- **S2:** it beats VGate-full (the pre-read gate of v7) with the interval above 0.
Reported with it: the share of questions on which the evidence answer is used, the gain over always-retrieve, the cost in
tokens relative to always-retrieve (it pays the closed-book pass as well), and the same rule under the fallback labels (where
"names no option" already falls back). Whatever the outcome it is reported; it is a result about answer selection on a
closed-set benchmark, not about read-or-skip gating.

**v7c outcome (2026-10-06 17:20; `experiments/results/cvqa_select__confirm.md`; held-out, n = 8,373, 95% intervals).**
The rule answers 67.2% of questions correctly (never-retrieve 65.6, always-retrieve 59.9, VGate-full 66.0, best of the two
answers 73.9). It uses the evidence answer on 33.1% of questions, keeps 71.9% of the cases where retrieval helps and avoids
68.8% of those where it harms, at 1.24 times the tokens of always-retrieve.
- **S1 held:** rule − better fixed policy +1.62 [+0.89, +2.42].
- **S2 held:** rule − VGate-full +1.24 [+0.59, +1.90]. (Rule − always-retrieve +7.26 [+6.50, +8.03].)
Development estimate +2.50 [+1.42, +3.66] (80%); the held-out value is lower and inside that interval's lower half.
Scope: one reader, one closed-set benchmark; the rule decides after reading and pays for both passes; on the open-ended
benchmarks' development splits the same rule is at or below the better fixed policy (see the diagnosis above). It is the first
registered held-out test in this project in which a per-query rule leads the better fixed policy by more than one point.

**Corrections (2026-10-06, after claim check 9).** (1) The clock times typed in the diagnosis, the v7c registration and its
outcome above (17:10 / 17:15 / 17:20) were ahead of the clock; the commits are at 16:49 (registration and script, before the
rule was applied to held-out answers) and 16:50 (outcome). (2) In the paper's vocabulary v7c is *registered*, not
preregistered: the held-out answers existed (J15, 16:33) when the rule was fixed on development data; only their comparison by
first-token probability had not been computed. The paper says "registered". (3) v8's count is reported in the paper for the
held-out split alone (five of six expectations held); the critic's three expectations are on development data, where its
question-level rule is ours and its scores existed before the one-sided amendment of X3.

# PREREG v6b — PILOT: re-ranking the top-20 with candidate page images (2026-10-06 17:20, before the rank 6–20 images are
complete and before any re-ranking is computed)
**Status: pilot on development data (E-VQA dev), no model pass; follows the diagnosis of v6 under the project's rule for
failed experiments.** What is known: among the five candidates, image similarity ranks the gold page first in 75.2% of cases
against 64.5% for the retriever; the gold page is among PreFLMR's top-5 / 10 / 20 passages for 60.5% / 70.2% / 78.7% of
questions; reading all five is right 72.4% of the time when the gold page is among them and 23.0% when it is not.
**Variant tested:** re-rank the twenty candidates by z(retriever) + w·z(image similarity) at page level (DINOv2-base, one lead
image per page; w ∈ {0.5, 1, 2, 4} chosen on four folds of image groups, applied to the fifth); the first five passages are
the new evidence. `scripts/kbimage_rerank.py`, committed with this text.
**Measured (zero GPU):** share of questions with the gold page, and the gold passage, among the five read; 80% image-cluster
bootstrap of the paired difference to the retriever's order.
**Go / no-go:** GO if the gold page among the five read rises by at least 6.0 points with the interval above +3.0. A GO leads
to a reader pass on the re-ranked evidence of the development split after the GPU quota reset (expectation to be registered
then: always-retrieve accuracy rises by at least 3 points). My estimate before running: about 55% for the GO (the image signal
adds 10.7 points of first-rank accuracy among five, but fifteen more distractor pages compete), about 30% for a 3-point
accuracy gain overall. A NO-GO covers this variant (one lead image, untrained fusion), not candidate images in general.

# PREREG v7d — CVQA on a second reader (2026-10-06 22:50, before any CVQA output of that reader exists)
**Why.** v7 and v7c rest on one reader (stated in the paper's limitations). The after-reading rule is the first per-query rule
of this project to lead the better fixed policy by more than a point on held-out data; a second reader tests whether that, and
the failure of the pre-read gate, are properties of the benchmark or of Qwen3-VL-4B.
**Run.** J18 (`kaggle/j18-cvqa-q25vl3b`, after the GPU quota reset): Qwen2.5-VL-3B-Instruct, closed-book and evidence passes on
all 10,374 questions with J14b's records, evidence (retrieved with the main reader's entity names; identical for both
readers), prompts and splits. Analyses unchanged: `scripts/cvqa_audit.py --model Qwen2.5-VL-3B-Instruct --run_dev J18
--run_confirm J18` (also `--fallback`) and `scripts/cvqa_select.py` with the same arguments; gates fitted on the development
split, applied once to the held-out split; 95% image-cluster bootstrap.
**Expectations (held-out split):**
- **R1′** retrieval helps on at least 8% and hurts on at least 8% of the development questions (if not, the rest is reported
  as a different regime and not as a replication).
- **G1′** as on the main reader, VGate-full does **not** lead the better fixed policy by one point (point estimate below 1.0).
- **S1′** the after-reading rule of v7c leads the better fixed policy by at least 1.0 point with the interval above 0.
- **S2′** it leads VGate-full with the interval above 0.
A replication family of its own; reported whatever the outcome. My estimate: S1′ about 70% (the rule has no parameter, but
Qwen2.5's first-token probabilities may be calibrated differently and its closed-book accuracy on CVQA is unknown to us).

**v7c, additions after claim check 10 (2026-10-06 23:00).** (1) The rule's registered form has two conditions (the evidence
reply names an option *and* is the more probable at its first token). Their parts on the held-out split: the probability
comparison alone leads never-retrieve by +0.76 [−0.01, +1.57]; on 3.0% of questions the evidence reply names no option and is
still the more confident one; both conditions together give the registered +1.62 [+0.89, +2.42]. (2) The registered "same rule
under the fallback labels" had not been computed; `scripts/cvqa_select.py --fallback` now does it
(`experiments/results/cvqa_select__confirm__fallback.md`). The rule itself is unchanged by those labels (it never uses a reply
that names no option); the gates gain from them, so like for like the rule leads VGate-full by +0.76 [+0.14, +1.44] (S2 still
holds) and VGate-post, not a registered comparison, by +0.55 [−0.08, +1.19] (registered labels: +1.24 and +1.03). S1 is
unchanged. Reading: the rule's lead over the better fixed policy stands; about half of its lead over the pre-read gates is the
discarding of replies that name no option, which a gate can be given too. (3) The rule was the best of four candidate rules
tried on development data (`experiments/results/cvqa_diag.md`, part 5). (4) Open-ended benchmarks, development split, four
readers: `scripts/select_open_ended_dev.py` → `experiments/results/select_open_ended__dev.md`; the rule is below the better
fixed policy in eleven of twelve reader-benchmark cells and level with it in one (main reader, InfoSeek, +0.3 [−0.5, +1.1]).
(5) The heading of v7d says 22:50; its commit is at 22:43.

# PREREG v9 — the three-number bound on what a gate can earn: prospective tests (2026-10-06 23:50, before J17 and J18 exist)
**Finding so far (exploratory; `scripts/gate_bound.py`, `experiments/results/gate_bound.md`).** A gate that reads a set of
queries gains h·TPR − m·FPR over never-retrieve (h, m: shares of queries that retrieval helps / harms; TPR, FPR: the shares of
those it reads), so a score's ceiling is set by h, m and its ROC curve for help against harm. With an equal-variance binormal
ROC of area A: G*(h, m, A) = max_x [h·Φ(d′ + Φ⁻¹(x)) − m·x] − max(0, h − m), d′ = √2·Φ⁻¹(A). On 148 held-out scores (three
benchmarks × four readers × 13 gates and signals) G* differs from the best gain each score reaches (threshold chosen with the
held-out labels) by 0.08 points on average (largest 0.71). To lead the better fixed policy by one point a score needs an
AUROC of 0.81–0.91 on those twelve benchmark–reader cells; the best of the 148 has 0.75. On CVQA the needed AUROC is 0.69:
the pre-read gates have 0.60–0.63 (bound +0.1 to +0.3, observed best +0.5 to +0.7) and the after-reading score 0.75 (bound
+1.75, observed best +1.68, registered rule +1.62). All of this was computed on data already analysed; the mathematics is
classical cost-sensitive ROC analysis.
**Prospective tests (no free parameter; the predictions follow from h, m and A of the new data):**
- **B1 (J17, ReflectiVA's retrieval token, three benchmarks, main reader's outcomes):** on each benchmark the gate's gain over
  the better fixed policy at its published decision is at most G*(h, m, A) + 0.3 points, with A the help-vs-harm AUROC of its
  two-token score; and it is below one point wherever A is below the needed AUROC.
- **B2 (J18, Qwen2.5-VL-3B on CVQA, held-out split):** for VGate-full, VGate-post and the after-reading score, the best gain the
  score reaches (threshold chosen with the held-out labels) is within 0.5 points of G*(h, m, A); the registered rules (û > 0;
  the v7c rule) earn at most G* + 0.3.
- **B3 (same run):** whether each of the three leads the better fixed policy by one point is predicted correctly by comparing
  its A with the needed AUROC, except within 0.02 of it (reported as undecided).
The script gets a `--prospective` mode before either run is analysed; outcomes are logged whatever they are.

# PREREG v9a — note on v9 after the literature check R14 (2026-10-07 00:14 IST, commit `55c6289`; J17 and J18 still do not exist)
**What was wrong in v9's wording.** G*(h, m, A) is not a bound. It is the lead of a score whose ROC curve for help against harm
is equal-variance binormal; a score with a cleaner top or bottom earns more. On the 148 held-out scores of v9 the best observed
lead exceeds G* for 80 (13 by more than 0.25 points; largest excess 0.71). The headline of v9 ("the three-number bound") and its
phrases "a score's ceiling", "bound +0.1 to +0.3", "to lead ... a score needs an AUROC of 0.81-0.91" are therefore to be read as
"estimate under a binormal curve". The bound that does hold: on a concave ROC curve TPR - FPR <= 2A - 1, so the lead over the
better fixed policy is at most min(h, m)(2A - 1) ("ceiling"); a one-point lead is out of reach below A = 0.5 + 0.01 / (2 min(h, m))
("floor"). Recomputed (`scripts/gate_bound.py`, `experiments/results/gate_bound.{json,md}`; exploratory, same data as v9): no
score exceeds the ceiling (0 of 148); the floor is 0.58-0.64 on the twelve cells and the best score is above it in 11; observed
leads are at 4% of the ceiling in the median (at most 48%); with sign and threshold chosen on half of the images and applied to
the other half, the scores lead by -0.01 points on average (largest 0.80; mean absolute difference to G* 0.10). Five of the six
published-prompt scores are below their benchmark's floor (with the sign that suits them). Library: `gate.gain_ceiling`,
`gate.gain_estimate` (was `gain_bound`), `gate.auroc_needed(shape="any" | "binormal")`.
**Prior work found by R14** (`docs/strengthening/R14_gate_bound_novelty.md`; abstract pages and the cited passages re-read by the
main session): the mathematics is cost-sensitive ROC analysis (Provost and Fawcett 2001; operating range, Drummond and Holte
2006); TARG (2511.09803) states the gain identity for retrieval gating without a ROC analysis; Bouchard (2605.06350) has an
identity between the gain of score-based over random escalation and two AUROC terms for LLM cascades; Bhatti and Vaddina
(2610.01535) simulate the signal accuracy a safety router needs against the best fixed model; Hauff et al. (ECIR 2010) ask how
strong a query-performance predictor must be before selective query expansion pays off. Claim allowed: the application to
retrieve-or-skip gating, the help-vs-harm reduction, the check on 148 scores, the planner. Not allowed: "first bound", "no gate can".
**The tests B1-B3 of v9 stand as written** (they test the estimate, with tolerances). Two clarifications, made before either run exists:
- B3's "leads the better fixed policy by one point" refers to the best lead the score reaches (the quantity of B2), not to the
  registered rule's lead.
- On the main reader's CVQA data (already analysed, so not a test) the part of B2 "the registered rules earn at most G* + 0.3"
  would have failed for VGate-post (rule +0.60, G* +0.26); a failure of that part on J18 would be no surprise and is reported as a failure.
**Added test B4 (ceiling, J17 and J18):** for ReflectiVA's two-token score on each benchmark and for VGate-full, VGate-post and the
after-reading score on J18, the best lead the score reaches is at most min(h, m)(2A - 1) with A its help-vs-harm AUROC (an
empirical ROC curve need not be concave, so this can fail).
`python scripts/gate_bound.py --prospective` computes B1-B4 and writes `experiments/results/gate_bound__prospective.{json,md}`;
it was dry-run on the main reader's CVQA runs (output not kept) and prints NOT_RUN for runs that do not exist.

# PREREG v10 — does a non-linear benefit model find the clean block that a linear one does not? (2026-10-07 00:19 IST, before the script exists)
**Why.** v9a: the gate scores are above the AUROC floor in 11 of 12 cells and still reach 4% of the ceiling in the median; their
help and harm cases overlap like two normal distributions. The ceiling is reached by a score that isolates a clean block of the
rarer outcome. A linear model cannot form conjunctions (a confident closed-book answer AND weak retrieval scores).
**The one variant tested.** Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`: depth 3, learning rate 0.05,
200 iterations, L2 1.0, no early stopping; three classes u in {-1, 0, +1}; score = P(+1) - P(-1)), fitted separately on each
benchmark of the main reader's held-out log (`demo/sample_log.csv`: its scalar signals and 64 hidden-state components), five
folds out of fold with image groups kept together. Comparator: logistic regression on standardised features in the same folds.
Rule: read iff the score is positive; nothing is tuned.
**Status of the data.** These held-out outcomes have been analysed many times: EXPLORATORY whatever the result.
My estimate before running: about 10% that the rule leads the better fixed policy by one point on at least one benchmark.
**Expectations.** T1: the boosted model's rule leads the better fixed policy by less than one point on each of the three
benchmarks. T2: its help-vs-harm AUROC is within 0.03 of the logistic model's on each.
**If T1 fails** (a lead of at least one point with a 95% image-cluster interval above zero on some benchmark): no claim yet; the
same model is run on the other three readers' held-out logs and on CVQA before anything goes into the paper.
**Not tested here** (my estimates of the chance of a one-point lead): monotone or rule-list models (about 5%); features from a
second sampling pass (about 15%; needs GPU); signals read after the evidence (about 50% where both outcomes are common; a
different cost regime; overlap guard).
Script: `scripts/tail_gate_check.py` -> `experiments/results/tail_gate_check.{json,md}`.

## Outcome log — v10 (computed 2026-10-07 00:21 IST, `experiments/results/tail_gate_check.{json,md}`; exploratory)
**T1 NOT held, T2 NOT held; my 10% estimate was wrong about InfoSeek.** Rule's lead over the better fixed policy, logistic /
boosted: E-VQA -0.18 [-0.59, +0.17] (AUROC 0.718) / -0.18 [-0.57, +0.19] (AUROC 0.734); InfoSeek +1.21 [+0.74, +1.76] (AUROC 0.779) / +1.46 [+0.89, +2.06] (AUROC 0.857);
OK-VQA -0.17 [-0.52, +0.17] (AUROC 0.65) / -0.15 [-0.44, +0.12] (AUROC 0.634). On InfoSeek both models lead always-retrieve by more than one point
with intervals above zero, and the logistic model does too: the difference to the paper's gates (largest lead 0.72) is therefore
not the model class. What differs: these models are fitted on InfoSeek alone, on about 2,900 held-out questions per fold (the
paper's gates: 1,000 development questions per benchmark, fitted pooled), and the folds are grouped by image only.
**No claim is made from this.** It contradicts the sentence "none of our read-or-skip gates beats the better fixed policy by one
point" for a gate fitted this way, so it has to be understood before the paper is touched.
**Diagnosis specified now, after the result (exploratory):** InfoSeek's 4,637 questions use 408 distinct question strings (one
template up to 123 times), so a model fitted on InfoSeek's log can learn which templates retrieval helps. `scripts/tail_gate_diag.py`:
(a) the same models with folds that keep every question string together; (b) a policy that is told the question string and
applies its better fixed policy as estimated on the training folds (unseen strings: the benchmark's better fixed policy);
(c) the same fit with 1,000 training questions per fold; on all three benchmarks. Reading: if (a) removes the lead and (b)
matches the gate, the lead is template identity (the audit's finding one level down), not per-query prediction. The registered
replication on the other readers and CVQA follows after that.

## Outcome log — v10, diagnosis and the registered replication on the other readers and CVQA (2026-10-07 00:36 IST; exploratory;
## `scripts/tail_gate_diag.py`, `experiments/results/tail_gate_diag*.{json,md}`)
Lead of the rule "read iff P(help) - P(harm) > 0" over the benchmark's better fixed policy, points, 95% image-cluster intervals;
one gate per benchmark, fitted out of fold on that benchmark's held-out log (image folds), and the same with folds that keep
every question string together (string folds); "told the string" applies the string's better fixed policy from the training folds.

| reader | benchmark | logistic, image folds | boosted, image folds | logistic, string folds | boosted, string folds | told the string | boosted AUROC, image -> string folds |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | evqa | -0.18 [-0.59, +0.17] | -0.18 [-0.57, +0.19] | -0.80 [-1.28, -0.39] | -0.51 [-0.88, -0.14] | -0.73 [-1.30, -0.18] | 0.734 -> 0.742 |
| Qwen3-VL-4B-Instruct | infoseek | +1.21 [+0.74, +1.76] | +1.46 [+0.89, +2.06] | -0.25 [-0.90, +0.40] | +0.08 [-0.30, +0.48] | +0.77 [+0.00, +1.53] | 0.857 -> 0.68 |
| Qwen3-VL-4B-Instruct | okvqa | -0.17 [-0.52, +0.17] | -0.15 [-0.44, +0.12] | +0.20 [-0.07, +0.47] | +0.00 [-0.32, +0.30] | -0.07 [-0.17, +0.00] | 0.634 -> 0.621 |
| Qwen2.5-VL-3B-Instruct | evqa | +0.00 [-0.43, +0.44] | +0.04 [-0.33, +0.42] | +0.18 [-0.25, +0.63] | +0.00 [-0.37, +0.40] | -0.98 [-1.55, -0.41] | 0.72 -> 0.736 |
| Qwen2.5-VL-3B-Instruct | infoseek | -0.19 [-0.78, +0.33] | +0.47 [-0.08, +1.01] | -1.46 [-2.08, -0.86] | -0.77 [-1.25, -0.29] | +0.11 [-0.58, +0.81] | 0.761 -> 0.514 |
| Qwen2.5-VL-3B-Instruct | okvqa | -0.37 [-0.79, +0.02] | -0.25 [-0.67, +0.15] | -0.10 [-0.49, +0.30] | -0.02 [-0.42, +0.35] | -0.05 [-0.12, +0.00] | 0.619 -> 0.626 |
| InternVL3_5-4B-HF | evqa | -0.51 [-0.92, -0.11] | -0.15 [-0.51, +0.19] | -0.18 [-0.51, +0.15] | +0.25 [-0.11, +0.61] | -1.24 [-1.81, -0.75] | 0.744 -> 0.771 |
| InternVL3_5-4B-HF | infoseek | +0.08 [-0.39, +0.58] | +0.77 [+0.37, +1.19] | -0.71 [-1.25, -0.14] | -0.30 [-0.73, +0.13] | -0.93 [-1.60, -0.29] | 0.769 -> 0.634 |
| InternVL3_5-4B-HF | okvqa | +0.67 [+0.15, +1.21] | +0.67 [+0.20, +1.14] | +0.87 [+0.32, +1.39] | +0.82 [+0.30, +1.31] | -0.02 [-0.07, +0.00] | 0.755 -> 0.77 |
| Qwen3-VL-8B-Instruct | evqa | +0.00 [-0.50, +0.48] | +0.04 [-0.37, +0.46] | +0.11 [-0.35, +0.57] | -0.18 [-0.61, +0.20] | -0.95 [-1.46, -0.45] | 0.729 -> 0.724 |
| Qwen3-VL-8B-Instruct | infoseek | -0.08 [-0.46, +0.32] | +0.47 [+0.08, +0.89] | -1.21 [-1.70, -0.74] | -0.77 [-1.20, -0.36] | -0.55 [-1.19, +0.08] | 0.81 -> 0.649 |
| Qwen3-VL-8B-Instruct | okvqa | -0.35 [-0.64, -0.02] | -0.44 [-0.74, -0.15] | -0.40 [-0.69, -0.10] | -0.27 [-0.52, -0.02] | -0.02 [-0.07, +0.00] | 0.617 -> 0.605 |
| Qwen3-VL-4B-Instruct | cvqa | +0.84 [+0.28, +1.36] | +0.48 [-0.06, +0.99] | +0.56 [-0.04, +1.09] | +0.24 [-0.30, +0.76] | -0.08 [-0.18, +0.00] | 0.621 -> 0.605 |

**What it shows.** (1) A lead of at least one point with an interval above zero in 2 of 24 fits on the three
benchmarks (Qwen3-VL-4B-Instruct / infoseek); an interval above zero in 6 of 24
(InternVL3_5-4B-HF / infoseek / boosted, InternVL3_5-4B-HF / okvqa / boosted, InternVL3_5-4B-HF / okvqa / logistic, Qwen3-VL-4B / infoseek / boosted, Qwen3-VL-4B / infoseek / logistic, Qwen3-VL-8B / infoseek / boosted). So v10's T1 failure
does not replicate as a one-point lead on the other three readers. (2) On InfoSeek, whose 3,637 held-out questions use 390 question
strings, every lead is at or below +0.08 once whole strings are held out, on all four readers, and the boosted model's
help-vs-harm AUROC falls by 0.14-0.25: what a gate learns on InfoSeek's log does not carry over to question strings that are not
in the log. (3) "Told the string" matches the gate only on the main reader (+0.77); on the other readers it is at or below the
fixed policy, so the gate is not simply a per-string fixed policy. (4) On E-VQA, OK-VQA and CVQA, whose questions rarely repeat,
image folds and string folds agree. (5) CVQA with about 6,700 labels per fold: logistic +0.84 [+0.28, +1.36],
below the one-point bar of v7 like the development-fitted gate. (6) With 1,000 labels per fold the main reader's InfoSeek lead is
+0.41 [-0.14, +0.99] (logistic) and
+1.04 [+0.56, +1.51] (boosted).
**Consequence for the paper.** The sentence "none of our read-or-skip gates beats a benchmark's better fixed policy by one point"
is true of the gates fitted on development data (1,000 questions per benchmark) and is restricted to them; the per-benchmark
fits above are reported next to it, with the string-fold result. Everything here is exploratory.
**Tool.** `gate.audit` gets a row for a gate fitted per segment and a mode for many small segments (question types): per-segment
policies from the training folds, unseen-segment gates on folds that keep whole segments out (commits `39b9a2d`, `07cef05`).

## Note on the v10 diagnosis — the string folds depend on the fold assignment (2026-10-07 01:02 IST; exploratory)
Running the audit's many-segments mode on InfoSeek's log (main reader) gave an unseen-question-string lead of -0.9 where
`tail_gate_diag.py` had -0.25 for the same model class: the two use different assignments of the 390 strings to five folds, and
a few strings hold 100 or more questions. Checked before relying on the range reported above: over ten random assignments (main
reader, InfoSeek; scratch run) the logistic lead is -0.56 on average (-1.10 to -0.05; AUROC 0.603) and the boosted lead -0.35
(-1.02 to +0.08; AUROC 0.656). The conclusion (no lead on question strings that are not in the log) does not depend on the
assignment; the single-assignment range in the table above does. `tail_gate_diag.py` now adds the ten-assignment mean and
range for every reader (`string_folds_repeated`), and the paper will quote those instead of one assignment.

## Outcome log — v6b (top-20 re-ranking with candidate page images; computed 2026-10-07 01:33 IST on the complete image set;
## `experiments/results/kbimage_rerank.{json,md}`; pilot on development data, no model pass)
Images: 7,811 of the 8,053 candidate pages have a lead image and all 7,811 are on disk (candidates with an image: 97.3%).
Weight chosen on every training fold: 1.0. E-VQA dev, n = 1000, 80% image-cluster intervals, % of questions:
gold page among the five read 60.5 -> 70.9 (+10.4 [+8.8, +12.0]); gold passage among the five read 57.7 -> 68.0 (+10.3 [+8.8, +11.8]); gold page ranked first
39.0 -> 50.2 (+11.2 [+8.8, +13.4]); gold passage ranked first 34.9 -> 44.9 (+10.0 [+7.9, +12.0]).
**GO by the registered criterion** (at least +6.0 with the interval above +3.0). My estimate had been about 55%.
**What this is and is not.** It is a gain in which page is read, for one variant (one lead image per page, DINOv2-base, untrained
sum of z-scores), on development data. It is not an accuracy result: no reader has seen the re-ranked evidence.
**Before the reader pass: a check specified now, after the result** (`scripts/kbimage_rerank_diag.py`): 413 of the 1,000
development photographs are Google Landmarks images, which are Wikimedia Commons images, so a page's lead image can be a near
copy of the question's photograph. The check reports the gain by source of the photograph, how many gold lead images have a
cosine above 0.9 / 0.95 to the photograph, and the gain when such near copies are not allowed to help. If the gain is carried
by near copies, the result describes this benchmark's images, not image matching for a user's own photograph, and is reported
that way.
**Next (registered in v6b):** a reader pass on the re-ranked evidence of the development split, with its expectation registered
before the job (always-retrieve accuracy up by at least 3 points).

## Outcome log — v6b, the near-copy check (2026-10-07 01:46 IST; `experiments/results/kbimage_rerank_diag.{json,md}`; exploratory)
Gold page among the five read, retriever order -> re-ranked, 80% intervals. All 1,000: 60.5 -> 70.9 (+10.4 [+8.8, +12.0]); with
near copies (cosine above 0.9 to the photograph) not allowed to help: 60.5 -> 70.6 (+10.1 [+8.4, +11.7]).
iNaturalist photographs (n = 587): 51.8 -> 64.2 (+12.4 [+10.4, +14.6]), and none of their gold lead images is above 0.9.
Google Landmarks photographs (n = 413): 72.9 -> 80.4 (+7.5 [+5.1, +10.0]); without near copies
72.9 -> 79.7 (+6.8 [+4.2, +9.3]); 8.0% of their gold lead images are
above 0.9. **The gain is not carried by near copies**; it is larger on the photographs that have none.
Extrapolation (not a measurement): the main reader is right on 72.4% of development questions with the gold
page among the five and on 23.0% without; at those rates the re-ranking would be worth +5.1
points of always-retrieve accuracy. The questions that gain the gold page are harder than average, so less is expected.

# PREREG v6c — PILOT: do the readers answer better with the image-re-ranked evidence? (2026-10-07 01:46 IST, before job J19 is pushed)
**Status: pilot on development data (E-VQA dev, 1,000 questions); never a confirmatory claim.**
**Run (J19, `kaggle/j19-evqa-rerank`).** Qwen3-VL-4B-Instruct and Qwen2.5-VL-3B-Instruct, the evidence pass only, same prompt and
budget (five passages, 160 words each), evidence = the five passages that `scripts/kbimage_rerank_export.py` puts first
(w = 1; evidence differs from the retriever's top-5 for 985 of the 1,000 questions; gold passage among the five: 68.0% against
57.7%). Scored with the benchmark's metric as every other run; compared, paired by question, with the same reader's answers on
the retriever's own top-5 (J1, J2) and with its closed-book answers.
**Expectations.** E1 (the GO of v6b): the main reader's always-retrieve accuracy rises by at least 3.0 points, 80%
image-cluster interval above zero. E2: the second reader's rises by at least 2.0 points. E3: the main reader's harm rate
(closed-book right, evidence answer wrong) does not rise by more than 1.0 point. My estimates before the run: E1 about 60%,
E2 about 60%, E3 about 75%.
**Also reported, no expectation:** the help and harm rates with the re-ranked evidence and what the planner makes of them
(floor, needed AUROC); the gain by source of the photograph.
**If E1 holds:** the same re-ranking on the held-out split needs lead images of its candidate pages (about 18,000 more; a
rate-limited download of several hours) and a held-out reader pass after the quota reset on 2026-10-10, registered before
it runs. Until then the paper may carry one sentence labelled as a development-data pilot.
**If E1 fails:** the result covers this variant (one lead image per page, DINOv2-base, untrained sum of z-scores, five
passages read) and is reported with the per-question loss located (gold page gained but answer still wrong, or a formerly
right answer lost).
**Not claimed either way:** that matching the photograph to page images is new for knowledge-based VQA (image-to-image
retrieval against Wikipedia images is part of published systems; `docs/strengthening/R10_entity_verification_rerank.md`). The
point for this paper is the comparison of what a better evidence order earns with what a gate earns.

# PREREG v10b — a boosted gate fitted on development data, the paper's own protocol (2026-10-07 01:49 IST, before the script exists;
# NOT blind, see below; exploratory)
**Why.** Claim check 13 (`docs/experiments/reviewer_simulation_13.md`, M1): the paragraph written after v10 changed three things at
once (model class, number of labels, the data fitted on) and named the labels. The development-fitted gates of the paper are
logistic and already one per benchmark; a boosted gate was never fitted on development data; and v10's own control with 1,000
labels separates the two model classes (InfoSeek, main reader: logistic +0.41 [-0.14, +0.99], boosted +1.04 [+0.56, +1.51]).
**Not blind.** The claim checker ran this configuration read-only and reported its numbers to me (InfoSeek, boosted, VGate-full's
features: main reader +1.04 [+0.61, +1.50], the other readers +0.49, +0.33, -0.16; E-VQA and OK-VQA -0.36 to +0.25; the lead on
held-out questions whose string occurs in the development set +1.46, on the others +0.00). This registration puts that
computation into a script with a database row; it is a reproduction, and the held-out outcomes have been analysed many times.
**The variant.** Per reader and benchmark: VGate-full's own pipeline (`operating_point.fit_benefit_gates`: the free and
retriever-score features plus 64 components of the closed-book hidden state, scaler and PCA fitted on that benchmark's 1,000
development questions), with the logistic model of the paper and with v10's boosted trees in its place (same hyperparameters);
fitted on the development questions, applied once to the held-out questions; rule: read iff the score is positive. Four
readers on E-VQA, InfoSeek, OK-VQA; the main reader on CVQA (2,001 development questions).
**Reported.** Lead over the better fixed policy with a 95% image-cluster interval, help-vs-harm AUROC, the planner's estimate;
and the same lead on the held-out questions whose question string occurs in the development set and on those whose string does
not, with the share of each.
**Expectations (reproduction).** R1: the logistic model reproduces the per-benchmark leads already in
`operating_point__*.json` (`per_dataset_gates`). R2: the boosted gate leads always-retrieve on InfoSeek by at least one point
with an interval above zero for the main reader and for no other reader. R3: on InfoSeek the boosted gate's lead on questions
with an unseen string has an interval that includes zero for every reader. R4: on E-VQA, OK-VQA and CVQA no boosted gate leads
by one point.
**Consequence for the paper, fixed now:** "no gate fitted on development data beats the better fixed policy by one point" is
said of the logistic gates only; if R2 holds the paper says that a boosted gate does on InfoSeek for the main reader, fitted
on development data, and reports the seen / unseen split; if R2 fails the paper says that the development-fitted boosted gate
does not and that the lead of v10 needs the held-out log.
Script: `scripts/tail_gate_dev.py` -> `experiments/results/tail_gate_dev.{json,md}`.

## Outcome log — v10b (computed 2026-10-07 01:55 IST; `experiments/results/tail_gate_dev.{json,md}`; exploratory, not blind)
**R1 held:** the logistic model reproduces `per_dataset_gates.benefit_rule_full` of `operating_point__*.json` in all twelve
reader-benchmark cells. **R2 held:** the boosted gate fitted on InfoSeek's 1,000 development questions leads always-retrieve on
the held-out questions by +1.04 [+0.61, +1.50] for the main reader (AUROC 0.769; the logistic gate: +0.41 [-0.21, +1.01]);
the other readers: Qwen2.5-VL-3B +0.49 [+0.05, +0.97]; InternVL3_5-4B-HF +0.33 [-0.11, +0.83]; Qwen3-VL-8B -0.16 [-0.56, +0.19]. **R3 held:** on held-out InfoSeek questions
whose string does not occur in the development set (11.4% of them) the boosted gate's lead is -0.24 [-0.77, +0.00]; -0.96 [-2.58, +0.69]; -0.24 [-1.53, +0.98]; +0.48 [-0.48, +1.50]
(readers in the order above, main reader first); on the 88.6% with a seen string the main reader's is +1.21 [+0.71, +1.73]. **R4 held:** no boosted gate
leads by one point on E-VQA, OK-VQA or CVQA (largest lead of any other fit in the file, either model: +0.49).
**Reading.** The exception to "no gate beats the better fixed policy by one point" is one model class on one benchmark for one
reader, whichever data it is fitted on (development: +1.04; held-out log, out of fold: +1.46), and it sits on question strings
the gate was fitted on. The logistic gate gets there only with the larger log (+1.21; with 1,000 labels +0.41). Label count is
therefore not the explanation that the paper's paragraph gave; the paragraph is rewritten.
**Share of held-out questions whose string occurs in the development set:** InfoSeek 88.6%, E-VQA 21.8%, CVQA 9.8%, OK-VQA 5.5%.

## Outcome log — v6c (J19, finished 02:12; analysed 2026-10-07 02:16 IST; `experiments/results/kbimage_reader_pass.{json,md}`;
## pilot on development data, 80% image-cluster intervals)
Run check: build stamp matched, smoke passed, 1,000 answers per reader, no error in either log; the answers on the retriever's
own five and the closed-book answers were re-scored in the same pass (always-retrieve 52.9% for the main reader, as before).
**E1 held:** main reader, always-retrieve accuracy 52.9% -> 59.3%, +6.40 [+4.99, +7.76].
**E2 held:** Qwen2.5-VL-3B, 48.5% -> 54.3%, +5.80 [+4.29, +7.28].
**E3 held:** the main reader's harm rate 5.5% -> 4.3% (-1.20 [-1.85, -0.58]); help 31.7% -> 36.9%.
By photograph (main reader): iNaturalist +5.45 [+3.69, +7.22], Google Landmarks +7.75 [+5.36, +10.02].
Where the gain is (main reader; share answered correctly with the retriever's five -> with the re-ranked five): gold passage
gained (n = 118) 24.6 -> 56.8; in both (n = 562)
74.4 -> 78.8; lost (n = 15) 60.0 -> 40.0;
in neither (n = 305) 23.9 -> 25.2. The extrapolation of
v6b (+5.1) was below the measured gain because answers also improve where the gold passage was already among the five.
Planner on the new rates: the AUROC a gate would need for a one-point lead rises from 0.857 to 0.9
(floor 0.591 -> 0.616): with better evidence there is less for a gate to earn.
**What this is.** A pilot on development data, for one benchmark and one variant; it supports no confirmatory claim. The technique
(matching the photograph to page images) is not new. **Next, as registered:** the held-out split, which needs the lead images of
its candidate pages, and a held-out reader pass registered before it runs.

# PREREG v6d — the image re-ranking on the held-out split (2026-10-07 02:25 IST; before the held-out candidate images are complete,
# before any held-out re-ranking is computed and before any reader output on re-ranked held-out evidence exists)
**Status: preregistered for the held-out split of E-VQA (2,750 questions).** What exists already for these questions: the
closed-book answers and the answers on the retriever's own top-5 (J4, J2), analysed many times. What does not exist: the lead
images of 5,384 of their 12,127 candidate pages (download in progress), any re-ranking of their candidates, any answer on
re-ranked evidence. Nothing of the method is changed after the development pilots (v6b, v6c): DINOv2-base, one lead image per
page, page score = z(retriever) + 1.0·z(image similarity) within the question's twenty candidates (a page without an image
gets the question's mean similarity), the first five passages are read; same prompt and budget; no threshold, nothing fitted.
**Run.** `scripts/kbimage_rerank_export.py --split confirm` (writes the evidence and its statistics), job J20
(`kaggle/j20-evqa-rerank-confirm`: Qwen3-VL-4B-Instruct and Qwen2.5-VL-3B-Instruct, the evidence pass),
`scripts/kbimage_reader_pass.py --split confirm` (95% image-cluster intervals; all answers scored in one pass).
**Precondition (not a result):** at least 95% of the candidates' pages that have a lead image on Wikipedia are on disk.
**Tests.** H1: the share of questions with the gold page among the five read rises by at least 6.0 points. H2: the main
reader's always-retrieve accuracy rises by at least 3.0 points, with the 95% interval above zero. H3: the second reader's
rises by at least 2.0 points, with the 95% interval above zero. H4: the main reader's harm rate does not rise by more than
1.0 point. My estimates: H1 85%, H2 80%, H3 75%, H4 85% (development: +10.4, +6.4, +5.8, harm −1.2).
**Reported whatever the outcome:** the gain by source of the photograph; help and harm rates and the planner's floor and needed
AUROC with the re-ranked evidence; where the gain and the loss sit (gold passage gained, lost, in both, in neither).
**Consequence for the paper, fixed now.** If H2 holds, the paragraph "Better evidence earns more" quotes the held-out numbers
as preregistered and drops the pilot label; if it fails, the paragraph says that the development gain did not hold on
held-out data and gives both. Either way the technique is described as known, and the claim is the comparison with what a
gate earns.

## Note on v9 / v9a — the number of scores (2026-10-07 03:52 IST)
The fresh-clone check of commit `1ead135` regenerated `gate_bound.json` with **156** scores, not 148: in the working tree four
intermediate files (`gate_scores__…__confirm.csv` of Qwen2.5-VL-3B and InternVL3.5-4B on E-VQA and InfoSeek) predated
`make extend` and lacked two scores each (`vgate_full`, `probe4`). The figures of v9 and v9a were computed on the incomplete
set. On the complete set (working tree regenerated, byte-identical to the fresh clone): 156 scores; mean absolute difference
of the estimate to the best observed lead 0.08 (largest 0.71); 86 leads above the estimate
(15 by more than 0.25 points); none above the ceiling; split-half mean lead -0.02;
floor 0.58–0.64, needed 0.81–0.91, best AUROC 0.748. No conclusion changes. The tests B1–B4 are unaffected
(they concern J17 and J18).


# PREREG v6e — which decision is worth a gate once the evidence is re-ranked (2026-10-07 13:06 IST; pilot on the E-VQA development split;
# no model pass; written before any per-question comparison of the J19 answers with the J1/J2 answers exists)
**Why.** User, 2026-10-07: do not drop what failed; go back, find why, and test what the mathematics says could work. Why our
read-or-skip gates do not lead by a point is known (v9, v9a): the lead is at most min(h, m)(2A − 1), the rarer outcome is about
5% of questions, and no score ranks help against harm above 0.75. Two things change with the image re-ranking of v6b/v6c and
were not tested: (i) a new signal exists before reading, the similarity of the photograph to the candidate pages' lead images,
which marks whether the right page was found; (ii) a second decision exists, *keep the retriever's five or read the re-ranked
five*, with its own help and harm shares.
**What exists.** J19 (both readers' answers on the re-ranked five, development split), scored in aggregate only (v6c: +6.4 and
+5.8 points; help and harm against the closed-book answer). Not computed: the per-question transitions between the retriever's
five and the re-ranked five; any gate that uses the image similarity.
**Analysis** (`scripts/rerank_gate_pilot.py` → `experiments/results/rerank_gate_pilot.{json,md}`). Three decisions, labels from
one scoring pass: A skip or read the retriever's five (the known case, for reference); B skip or read the re-ranked five; C read
the retriever's five or the re-ranked five. For each: h, m, the planner's AUROC floor and needed AUROC for a one-point lead, and
gates fitted out of fold (five folds by image): features F0 = VGate-full's (the paper's signals and 64 components of the
closed-book hidden state, scaler and PCA fitted in fold) and F1 = F0 plus eight image-similarity features that need no label
(highest similarity among the twenty; similarity of the retriever's first page and of the re-ranked first page; margin of the
combined score between the first two pages; standard deviation of the similarities; share of candidates with an image; how many
of the five read change; z-score of the retriever's first page); models: the paper's logistic model and the boosted trees of
v10. Rule: take the action iff P(help) − P(harm) > 0. Reported: AUROC, lead over the better fixed policy with an 80%
image-cluster interval, the planner's estimate and ceiling. Sixteen fits per reader beyond the reference; a pilot, no test.
**Expectations, written before the numbers.** B: the harm share falls with re-ranking (5.5% → 4.3% in v6c), so the needed AUROC
rises (0.86 → 0.90); the image features should raise the AUROC, but not by 0.15. Probability that any B fit leads by a point
with the interval above zero: 10%. C: the re-ranking mostly helps; I expect the harm share of switching to be 3–5% and much of
it reader noise on questions where the gold page is among both fives, which no signal predicts. Probability that any C fit
leads always-re-rank by a point with the interval above zero: 10–15%.
**GO rule.** If a B or C fit of the main reader leads by at least 1.0 point with the 80% interval above zero and the same fit
has a positive lead on the second reader, that one fit is registered for the held-out split before J20's outputs exist.
Otherwise the planner's numbers for B and C are reported as what they are: the planner applied to two further decisions.

## Outcome log — v6e (computed 2026-10-07 13:16 IST; `experiments/results/rerank_gate_pilot.{json,md}`; pilot on development data, 80% intervals)
**Variant tested.** Sixteen fits per reader: decisions B (skip or read the re-ranked five) and C (the retriever's five or the
re-ranked five) × features F0 / F1 × logistic / boosted, out of fold on 1,000 development questions of E-VQA, rule
"take the action iff P(help) − P(harm) > 0". **GO: none.** No fit leads the better fixed policy; the largest lead is −0.30
(B) and −0.40 (C) on the main reader. Both expectations held (I gave a lead 10% and 10–15%).
**The planner's numbers** (main reader / second reader). A: helps 31.7 / 30.6%, harms 5.5 / 6.7%, AUROC floor 0.59 / 0.58,
needed 0.86 / 0.82, best fit 0.74 / 0.72. B: helps 36.9 / 36.0%, harms 4.3 / 6.3%, floor 0.62 / 0.58, needed 0.90 / 0.85, best
fit 0.68 / 0.74. C: helps 9.5 / 9.6%, harms 3.1 / 3.8%, floor 0.66 / 0.63, needed 0.87 / 0.82, best fit 0.60 / 0.60. An oracle
gate would add 4.3 / 6.3 points (B) and 3.1 / 3.8 (C); switching every question to the re-ranked five adds 6.4 / 5.8.
**Mechanism, from the data.** (1) The image-similarity features do not rank help against harm: F1 changes the AUROC by −0.01 to
+0.02 in every cell. They mark whether the right page was found, and the re-ranking has already used that. (2) Decision C: on
the main reader 26 of the 31 questions the switch harms are questions where it did not change whether the gold passage is among
the five (16 with it in both fives, 10 with it in neither); only 3 lost the gold passage. On the second reader 28 of 38. So
the harm of switching is mostly the reader answering differently on a reordered or partly replaced context, which nothing
known before reading predicts: every C fit has an AUROC (0.54–0.60) below the floor (0.63–0.66), where no threshold can lead
by a point on a concave curve.
**Follow-up tested after the diagnosis (exploratory, not in the registration above).** If the harm of C is visible only after
reading, read both fives and answer with one of the two replies by their token log-probabilities (twice the evidence tokens).
"Re-ranked unless the other reply is more confident": −0.30 [−1.11, +0.51] (mean log-probability, AUROC 0.74) on the main
reader, −1.30 on the second. A logistic model on the six confidences, out of fold: +0.60 [+0.10, +1.11] on the main reader
(AUROC 0.72), −0.70 [−1.26, −0.10] on the second. Below a point and not replicated.
**Not tested, with my estimate.** (a) A larger candidate pool or learned fusion weights for the re-ranking (the gold page is
in the top-50 for 84% of development questions, in the top-20 for 75%, among the re-ranked five for 68%): improves the
evidence, not a gate; about 40% for two further points of always-retrieve accuracy; needs the lead images of roughly 30,000
more pages (several hours of rate-limited download), a new registration and two reader passes. Not before v6d is done.
(b) Reading the union of both fives: 20% (the reader loses accuracy with more and truncated passages, v3a); two reader passes.
(c) A gate for B or C on traffic where the rarer outcome is common (as on CVQA, 8% / 14%): no candidate images exist for CVQA.
**What this adds.** The planner, applied to two further decisions, said before any fit that neither is worth a gate (rarer
outcome 3–6% of questions, needed AUROC 0.82–0.90), and sixteen fits per reader agree. The held-out version of this statement
is registered next (v6f), before J20's outputs exist.

# PREREG v6f — the planner's forecast for two further decisions, on the held-out split (2026-10-07 13:20 IST; before the held-out candidate
# images are complete (9,928 of 11,778 at 13:20), before any held-out re-ranking is computed and before J20 exists)
**Status: preregistered for the held-out split of E-VQA (2,750 questions).** This is a forward test of the planner: from the
development split it says that neither decision B (skip or read the re-ranked five) nor decision C (the retriever's five or
the re-ranked five) is worth a gate (v6e). The held-out questions' answers on the re-ranked five do not exist yet.
**Run** (after v6d's own analysis): `scripts/rerank_gate_pilot.py --split confirm --images <dev images> --images_confirm
<held-out images>` → `experiments/results/rerank_gate_pilot__confirm.{json,md}`. The eight fits of v6e per decision and reader
(F0 / F1 × logistic / boosted), fitted once on all 1,000 development questions (scaler and PCA too) and applied unchanged to
the held-out questions, rule "take the action iff P(help) − P(harm) > 0"; 95% image-cluster intervals.
**Forecasts.** F1: for B and C and both readers, the held-out share of the rarer outcome is within 2.0 points of its
development share (B 4.3 / 6.3%, C 3.1 / 3.8%). F2: none of the sixteen B and C fits per reader leads the held-out better
fixed policy by 1.0 point or more (point estimate). F3: every C fit's held-out help-vs-harm AUROC is below the held-out AUROC
floor for a one-point lead. My estimates: F1 85%, F2 90%, F3 70%.
**Consequence for the paper, fixed now.** If F2 holds, the paragraph on the re-ranking may say, as preregistered, that the
planner forecast from development data that neither further decision was worth a gate and that no gate fitted for them led by
a point on held-out data. If F2 fails, the paper reports the fit that leads and that the planner's forecast was wrong for it.

# PREREG v11 — the audit on another group's released logs: Chen et al.'s knowledge-boundary gates with GPT-4o and
# Qwen-VL-Max as readers (2026-10-07 13:46 IST; registered: the logs are public, no answer of theirs has been scored by us)
**Why.** Two weaknesses the user listed on 2026-10-07: we audit two published *prompts* with our own small reader, not a
trained published gate inside its own system, and our readers are 3–8B. Research note R15 found that the repository of Chen
et al. (EMNLP 2025, arXiv:2502.18023; github.com/Chord-Chen-30/VLLM-KnowledgeBoundary, Apache-2.0) releases, for the same
questions, the closed-book and the search-augmented answers of five VLMs (GPT-4o, Qwen-VL-Max, Qwen-VL-2, Qwen-VL-Chat,
DeepSeek-VL-Chat) and the search decisions of their fine-tuned hard and soft knowledge-boundary models, of a prompt baseline
and of a human, with their scoring script. That is an outcome log with logged gates, which is the audit's input.
**What I have looked at.** File names, field names, the scoring code, the number of rows (Mix 600, Life 149), the order of
rows, the source folder of each image (Mix: six sources of 100 rows each, in blocks: the Life set, a second in-house set, two
Dyn-VQA-style sets in Chinese and English, NoCaps, Visual7W), and the counts of the hard model's decisions (Mix: 461 search,
139 not; Life: 144 and 5). **Not computed:** the accuracy of any answer, any decision by source, any policy.
**Analysis** (`scripts/chen_logs_score.py`: their accuracy function run unchanged on their files, the judge-model score left
out because it needs a paid API → `experiments/results/chen_logs__questions.csv`; `scripts/chen_logs_audit.py` →
`experiments/results/chen_logs_audit.{json,md}`). Mix set, each reader. Segment = the source of the question (block of 100).
Policies, with their graded accuracy: never search; always search; each logged gate (hard; soft at their default threshold
4.0; prompt; human); *told the source* = the better of never / always per source, chosen on four folds and applied to the fifth
(so the choice never sees the questions it is scored on); the logged gate's majority decision per source; an oracle gate.
Intervals: 95% bootstrap over questions, 2,000 resamples. Reproduction check first: never, always and the hard gate must
match the accuracies their paper prints for Mix where it prints them; a mismatch is reported and the analysis is labelled
accordingly. The planner per source for GPT-4o and Qwen-VL-Max (help / harm: the graded accuracy rises / falls across 0.5,
their own threshold for "correct").
**Expectations, fixed now.** K1: for GPT-4o and for Qwen-VL-Max, told-the-source is at least as accurate as the hard gate
(point estimate). K2: the same against the soft gate. K3: the hard gate's decisions agree with its own majority decision per
source on at least 85% of the Mix questions. K4: for at least four of the five readers the hard gate does not lead
told-the-source by a point with an interval above zero. My estimates: K1 60%, K2 60%, K3 75%, K4 75%.
**If they fail.** A gate that beats told-the-source on its own data is reported as exactly that: a published gate that, on
this log, learned more than the source. The paper states the outcome either way.
**Limits, known now.** 100 questions per source; one run; their retrieval was live web search; many questions are Chinese;
their graded token accuracy, not a judge; the source label is our segment choice (the only one tested).

## Outcome log — v11 (computed 2026-10-07 13:51 IST; `experiments/results/chen_logs_audit.{json,md}`, `chen_logs__questions.csv`; registered)
**Reproduction first.** Their accuracy function, run unchanged on their files, gives for the Mix set exactly what their Table 11
prints: GPT-4o never 34.52, always 35.96, hard gate 35.90 (searching 76.83%), soft gate 35.86 (49.33%); Qwen-VL-Max 32.35, 34.78,
35.48, 34.84. (For the Life set GPT-4o matches too; Qwen-VL-Max's closed-book accuracy is 25.42, which is the figure of their
Table 13, while Table 11 prints 24.21.) One soft-gate decision has a format their parser sets to "no search" with a warning.
**Disclosure.** For this check I printed never, always and the two gates by source before the cross-fitted policy was computed,
so the direction of K1 and K2 was visible to me a few minutes before the audit ran. The expectations were committed before
anything was scored (`cebfc9c`) and were not changed.
**Outcome.** K1 held, K2 held, K4 held, **K3 NOT held** (the hard gate agrees with its own majority decision per source on 82.8%
of the questions; I had required 85%). Told-the-source (cross-fitted) against their gates, accuracy difference in points, 95%
bootstrap over questions: hard gate − told: GPT-4o −2.38 [−3.92, −0.86], Qwen-VL-Max −0.87 [−2.23, +0.52], Qwen-VL-2 −0.38
[−1.66, +1.02], Qwen-VL-Chat −0.86 [−1.76, +0.06], DeepSeek-VL-Chat −0.35 [−1.48, +0.74]; soft gate − told: −2.42, −1.50, −1.04,
−2.61, −2.67 (the interval excludes zero for GPT-4o, Qwen-VL-Chat and DeepSeek-VL-Chat); the human's decisions − told: −1.80,
−0.84, −0.34, −2.58, −3.00. No gate of theirs is ahead of told-the-source for any reader. Told-the-source searches 54–70% of the
questions (by reader), their hard gate 76.8%, their soft gate 49.3%.
**What this does and does not show.** On this log their gates add no accuracy beyond knowing which of the six sources a
question comes from, for every reader including GPT-4o; for GPT-4o the hard gate is 2.4 points below that policy and equal to
always-search. It does not show that the gates are useless: the soft gate reaches always-search accuracy with half the searches
(their efficiency claim; told-the-source searches more than the soft gate for every reader), told-the-source needs the
source label, each source has 100 questions, the accuracy is their graded token overlap, and the search was live.
**Planner** (correct = accuracy above 0.5, their threshold): GPT-4o helps 7.5%, harms 6.8%, floor 0.57, needed 0.63; Qwen-VL-Max
11.0%, 7.3%, 0.57, 0.68. Retrieval is close to balanced for GPT-4o on this mixture (always − never = +1.4 points).
**Reproduction, all five readers (added after the outcome above; no number of the audit changes).** Against the Mix rows of
their Table 3 (Qwen-VL-Chat) and Table 11 (the others), six policies each (never, always, prompt, hard, soft, human): exact
for GPT-4o, Qwen-VL-Max, Qwen-VL-Chat and DeepSeek-VL-Chat; for Qwen-VL-2 the closed-book accuracy is 32.45 against their
printed 32.59 (and the prompt policy, which mostly answers closed-book, 33.13 against 33.27), the other four policies exact.

## Notes on v11 after claim check 14 (2026-10-07 14:15 IST; `docs/experiments/reviewer_simulation_14.md`; no registered number changes)
- **Scope.** The released decisions are those of the boundary models fine-tuned from Qwen-VL-7B-Chat; their paper reuses them
  as a "surrogate" for the other four readers, GPT-4o included. Their DeepSeek-based gates are not in the released Mix folder
  and are not audited. All six sources are outside the gates' training data, while told-the-source is chosen on about 80
  labelled questions per source: the same asymmetry as our own unseen-benchmark test, now stated in the paper.
- **Metric.** The audit uses their token accuracy only. Their main tables use an LLM-judge score, which needs a paid API and is
  not computed; under it their Table 4 prints the hard gate above always-search for GPT-4o (54.10 against 52.90). Whether
  told-the-source leads under the judge score is unknown. The paper now says "by their token accuracy".
- **"Human"** in the outcome log above is their boundary model trained on human-labelled data, not a person's decisions.
- **Fold assignments (added, exploratory).** Over 200 other five-fold assignments of told-the-source the hard gate − told
  averages −2.37 (GPT-4o), −1.00 (Qwen-VL-Max), −0.36 (Qwen-VL-2), −1.05 (Qwen-VL-Chat), −0.48 (DeepSeek-VL-Chat) and the hard
  gate is ahead in 0%, 1.5%, 5.0%, 0% and 8.5% of them; the soft gate is ahead in none.
- **Soft gate and always-search.** "Matches always-search with 49% of the searches" holds for GPT-4o (−0.10), Qwen-VL-Max and
  Qwen-VL-2 (+0.06, +0.07), not for Qwen-VL-Chat (−1.41) or DeepSeek-VL-Chat (−2.17); the paper attaches it to GPT-4o.
- **Order of commits.** The expectations are in `cebfc9c`; the scoring script and the scored table were first committed with
  the outcome (`1842306`), so git shows the expectations before the outcome but not before the scoring code. K1 and K2 name
  GPT-4o and Qwen-VL-Max; the statement for all five readers comes from the registered analysis, not from an expectation.
- **A label I could not confirm:** the fourth block's accuracies (GPT-4o 24.3 / 27.8) are far from their full Dyn-VQA (en) set
  (43.33 / 49.71). The audit needs the blocks to be the six sources (their search rates per block follow the paper's order),
  not their names.
- What their paper claims is fewer searches at maintained accuracy on held-out data; the audit contradicts no claim of theirs.

## Outcome log — v6d (J20 finished 15:47; analysed 2026-10-07 15:56 IST; `experiments/results/kbimage_reader_pass__confirm.{json,md}`;
## preregistered, held-out split of E-VQA, n = 2,750, 95% image-cluster intervals)
**Precondition met:** all 11,778 candidate pages that have a lead image are on disk (100%; 96.8% of the candidates per question
have an image). J20: build stamp as expected, 2,750 answers per reader, no error in the log.
**All four tests held.** H1: the gold page is among the five read for 64.3% → 71.4% of the questions (+7.1; the gold passage
60.0% → 67.5%). H2: the main reader's always-retrieve accuracy 51.2 → 55.6, **+4.44 [+3.07, +5.84]**. H3: Qwen2.5-VL-3B 48.6 →
52.8, **+4.22 [+2.87, +5.62]**. H4: the main reader's harm rate 4.7% → 4.1% (−0.62 [−1.16, −0.11]); the second reader's 4.7% →
4.3% (−0.36 [−0.91, +0.18]).
**Smaller than on development data** (+6.4 and +5.8 there; gold page +10.4 there). By source of the photograph: iNaturalist
+5.38 [+3.58, +7.14] and +5.52; landmarks +3.44 [+1.39, +5.62] and +2.84 [+0.87, +4.83] (on development data landmarks gained
more, +7.75). Where: questions that gain the gold passage (267) 28.1 → 58.4% right; that lose it (60) 55.0 → 31.7%; with it in
both fives (1,589) 70.9 → 74.0%; in neither (834) 20.9 → 21.5%.
**For comparison:** an oracle read-or-skip gate on the retriever's five would add 4.7 points for the main reader (its harm
share); the planner's AUROC floor / need for a one-point lead moves from 0.606 / 0.881 to 0.622 / 0.905 with the re-ranked five.

## Outcome log — v6f (computed 2026-10-07 15:56 IST; `experiments/results/rerank_gate_pilot__confirm.{json,md}`; preregistered)
**All three forecasts held.** F1: the rarer outcome's held-out share against its development share: B 4.1% (4.3%) and 4.3%
(6.3%; exactly the 2.0 points allowed); C 3.8% (3.1%) and 4.3% (3.8%). F2: none of the sixteen development-fitted B and C gates
per reader leads the held-out better fixed policy by a point; the largest lead is −0.15 on the main reader and +0.15 [−0.24,
+0.57] on the second (B, boosted, with the image features). F3: every C fit's held-out AUROC (0.49–0.59) is below the floor
(0.63 and 0.62). Where the switch harms: 83 of 105 questions (main reader) and 96 of 118 (second) kept the gold passage in both
fives or in neither, as on development data.
**Reading.** From 1,000 development questions the planner said that neither skipping the re-ranked evidence nor choosing
between the two evidence sets is worth a gate; on 2,750 held-out questions no gate fitted for either decision led by a point.
The untrained re-ranking itself adds about what a perfect read-or-skip gate would (4.4 against 4.7 points, main reader).

## Correction to v6e / v6f — the number of fits (2026-10-07 16:05 IST)
The entries above say "sixteen fits per reader" for decisions B and C. The count is **eight per reader** (two decisions × two
feature sets × two models), sixteen over the two readers; with the reference decision A, twelve per reader. Nothing else
changes: F2 was evaluated on all of them (`rerank_gate_pilot__confirm.json`), and the paper says "8 such gates per reader".

## Corrections after claim check 15 (2026-10-07 16:40 IST; `docs/experiments/reviewer_simulation_15.md`)
- **v6f, F1 does not hold on exact arithmetic.** For decision B on Qwen2.5-VL-3B the rarer outcome is 63 of 1,000 development
  questions (6.300%) and 118 of 2,750 held-out questions (4.291%): 2.009 points apart, where 2.0 were allowed. The registered
  script compares the shares rounded to one decimal (6.3 against 4.3) and so prints "held"; my outcome log's "exactly the 2.0
  points allowed" was wrong. Stated now as: **six of the seven tests of v6d and v6f held; F1 held only at the script's
  rounding.** The script reports both (`F1`, `F1_unrounded`); the paper says "six of seven tests held, the seventh only by
  rounding". F2 and F3, the forecasts about the gates, are unaffected.
- **v6e outcome log:** the main reader's largest B lead on development data is −0.40 (the log says −0.30).
- **Who forecast.** The planner gave a needed AUROC of 0.82–0.90 and estimates of at most +0.12 for these decisions; the
  forecasts F1–F3 and their probabilities were mine, written after the same eight gates per reader had failed out of fold on
  development data. The paper now says "we forecast with the planner".
- **"Untrained".** No parameter is learned and nothing was tuned on the held-out split, but the one fusion weight (1.0) was
  chosen on development folds with gold-page labels (v6b). The paper now says so.
- **H1's statistic** (gold page among the five read, 64.3% → 71.4%) is now also in the tracked
  `kbimage_reader_pass__confirm.json`; that file names the tests E1–E3, which are H2, H3 (without its interval condition, which
  holds: lower bound +2.87) and H4.
- **After-reading rule on the open-ended development splits:** below the better fixed policy in 11 of 12 reader–benchmark pairs
  by point estimate, in 10 with an 80% interval below zero; the paper had printed 10 without the interval level after a trim
  and now prints 11.
- The comparison with an oracle gate (4.4 against 4.7 points) is descriptive, not a registered test.

## Outcome log — v8, ReflectiVA's retrieval token (J17, run 2026-10-07 16:00–~19:15, analysed 2026-10-08 11:57 IST;
## `experiments/results/external_gates__{confirm,dev}.{json,md}`; preregistered expectations X1–X3, held-out split)
J17: 13,433 questions (development and held-out, three benchmarks), no error in the log; the two tokens carry 97–98% of the
first-token probability. **X1 held, X2 held, X3 NOT held.** The released checkpoint's first token is [Retrieval] for 100% of
the held-out E-VQA and InfoSeek questions and 78.7% of OK-VQA. Paired with the main reader's two answers: mixed-traffic
accuracy 40.64 (never 33.47, always 39.50, told the benchmark 46.05, oracle 50.79); gate − told the benchmark **−5.40 [−5.88,
−4.92]**; gate − always +1.15 [+0.94, +1.37]; gate − its own majority decision per benchmark +1.15 [+0.94, +1.37] (X3 required
this to be below 1.0: the token's skips on OK-VQA are worth more than a point over always-retrieve there, 43.4 against 39.9).
Help-vs-harm AUROC of its two-token score 0.51 / 0.47 / 0.53 (E-VQA / InfoSeek / OK-VQA), below the AUROC floor on each.
Exploratory, added after the result as for the other gates: against random decisions at its read rate on each benchmark the
gate is −0.25 [−0.44, −0.03] in mixed traffic (−0.75 [−1.36, −0.10] on OK-VQA): skipping as often at random does as well.
Development split: the same picture (gate − told −5.7 [−6.6, −4.7]; reads 100 / 100 / 80%).
**Limits.** The token is the ReflectiVA model's decision; the two answers it chooses between are our reader's, not
ReflectiVA's own. Held-out expectation count over the three gates audited this way: 7 of 9.

# PREREG v11b — the audit of Chen et al.'s logs under an LLM-judge score (2026-10-08 15:38 IST; registered: no judge score of any answer exists)
**Why.** Claim check 14 (M1): v11 uses their token accuracy, but their paper's main metric is an LLM-judge score (Qwen-Max
through a paid API), under which their Table 4 prints the hard gate above always-search for GPT-4o on Mix (54.10 against 52.90).
Whether told-the-source leads under a judge score is unknown, and the paper has to say "by their token accuracy".
**What is fixed now.** Their judge prompt, unchanged (`evaluation/src/vqa_eval.py`, `qwen_evaluate`: a system prompt asking for
a score between 0 and 1, then the processed model output and the reference answers; the question is not shown). The judge
inputs are captured from their own code path (`scripts/chen_logs_score.py --judge_inputs`). **Judge model:** Qwen-Max if the user
provides a DashScope key (their metric exactly); otherwise GPT-4o if an OpenAI key is provided (their Table 9 reports it within
3 points of Qwen-Max on Mix); otherwise the open Qwen2.5-14B-Instruct, 4-bit, greedy, on Kaggle after the quota reset (job
`kaggle/j21-chen-judge`). One judge only is run for the registered analysis; the choice follows that order and is recorded.
A reply that does not start with a number scores 0 (their code); scores are clipped to [0, 1].
**Validity gate (not a result).** Their Tables 3 and 4 print the Mix judge score of six policies for each of five readers
(30 numbers). The judge is *valid for this analysis* if its never-search and always-search means are within 5.0 points of the
printed ones for at least four of the five readers. If not, the judge analysis is reported as not validated and supports no
claim in the paper; the judge is not changed after seeing its scores.
**Analysis** (`scripts/chen_logs_audit.py --metric judge` → `experiments/results/chen_logs_audit__judge.{json,md}`): exactly v11's,
with the judge score in place of the token accuracy (told-the-source cross-fitted on the judge score; same folds and seeds).
**Expectations.** L1: for GPT-4o, told-the-source is at least as high as the hard gate (point estimate). L2: the same for at
least four of the five readers. L3: the hard gate does not lead told-the-source by a point with an interval above zero for
any reader. My estimates: validity gate 70% (open judge) / 90% (GPT-4o or Qwen-Max); L1 55%, L2 55%, L3 80%.
**Consequence for the paper, fixed now.** If the gate passes: the use-case paragraph reports both metrics; if L1 or L2 fail it
says that under the judge score their gate is ahead of told-the-source for those readers, and the abstract's sentence is
restricted or withdrawn accordingly. If the gate fails: the paper keeps "by their token accuracy; their judge score is not
computed" and says a substitute judge did not reproduce their scores.

## Outcome log — v9 / v9a, B1 and B4 on ReflectiVA's token (J17; computed 2026-10-08 16:10 IST;
## `experiments/results/gate_bound__prospective.{json,md}`; preregistered). B2, B3 and B4 on CVQA wait for J18.
**B1 held on all three benchmarks.** The token's lead over the better fixed policy at its published decision is 0.00 (E-VQA),
0.00 (InfoSeek) and −16.21 (OK-VQA); the estimate G* is 0.00 on each (help-vs-harm AUROC 0.509, 0.470, 0.525; needed 0.881,
0.823, 0.857), so "at most G* + 0.3" and "below one point where the AUROC is below the needed one" both hold.
**B4 held on two of three; NOT on InfoSeek, by 0.05 points.** Best lead any threshold reaches: 0.00, 0.05, 0.05; ceiling
min(h, m)(2A − 1): 0.08, 0.00, 0.25. On InfoSeek the score's AUROC is below one half (0.47), so the ceiling is zero, while a
threshold in the score's given direction still finds +0.05 points: its empirical ROC curve is not concave, the case the
registration named ("an empirical ROC curve need not be concave, so this can fail"). With the sign that suits the score, as the
exploratory analysis of the 156 scores treats every score, A is 0.53 and the ceiling 0.28. Mechanism, not excuse: the test as
registered failed in this cell.

## Outcome log — v11b (J21 on 2026-10-10, 11:10–11:50, two versions; analysed 2026-10-10 11:53 IST;
## `experiments/results/chen_logs_audit__judge.{json,md}`; judge: Qwen2.5-14B-Instruct, 4-bit, greedy — no key was available)
**Run.** Version 1 judged 5,992 of 6,000 answers: one worker ran out of memory on its last batch (the eight longest replies).
Version 2 judged those eight one at a time with the same judge, prompt and decoding. No reply failed to start with a number.
Nothing was scored before all 6,000 were in.
**Validity gate: NOT passed.** The judge's never-search and always-search means are 6.3–13.1 points below the printed ones for
every reader (GPT-4o 38.3 / 43.2 against 51.4 / 52.9; the gate allowed 5.0 for four of five). **So, as registered, this judge
supports no claim in the paper**, and the paper keeps "by their token accuracy; their judge score is not computed".
**Variant tested:** their prompt unchanged, Qwen2.5-14B-Instruct in 4-bit, greedy, their parsing; one judge, one run.
**Mechanism, from the data.** The open judge is uniformly stricter, not differently ordered: over the 30 printed policy means
(five readers × six policies) its means are 6.0–13.1 points lower (9.2 on average) and correlate with the printed ones at 0.96
(rank correlation 0.92). It gives a full score to 8.9% of the answers, zero to 31.7% and partial credit below one half to 30.3%.
The authors' own README says of this metric that it "might be different from the paper", so even their judge may not reproduce
the printed numbers.
**For the record, not a claim** (the registered expectations, computed because the script prints them): L1, L2, L3 held under
this judge (hard gate − told-the-source −0.95 [−1.85, −0.07] for GPT-4o; +0.04 for Qwen-VL-Max, negative for the other three).
**Not tested, with my estimate of passing the same gate.** (a) Qwen-Max with a DashScope key: 70% (the README's caveat keeps it
below certainty); (b) GPT-4o with an OpenAI key: 60%; (c) a larger open judge, Qwen2.5-32B-Instruct in 4-bit: 20–25%, about
three GPU-hours of the free quota. (c) is run next as its own registration; (a) and (b) need a key the user does not have.

# PREREG v11c — a second open judge for v11b (2026-10-10 11:53 IST; registered: no score of this judge exists)
The first open judge failed v11b's validity gate. One more judge is tried, **Qwen/Qwen2.5-32B-Instruct, 4-bit, greedy**
(job `kaggle/j22-chen-judge-32b`), with everything else of v11b unchanged: prompt, inputs, parsing, the validity gate (never
and always means within 5.0 points of the printed ones for at least four of five readers), the analysis, the expectations
L1–L3 and the consequences for the paper. It is a second attempt after a failure and is reported as such; the gate is judged
against the authors' printed scores, which do not depend on our audit. No third open judge will be tried. My estimate that the
gate passes: 20–25%. Output: `experiments/results/chen_logs_audit__judge32.{json,md}` (`--judge_run experiments/runs/J22
--tag judge32`).

## Note on v7d before its analysis — 72 failed generations in J18 (2026-10-10 13:09 IST; nothing of v7d has been computed)
J18 finished (10,374 questions, both passes). For **72 questions (0.7%; 18 development, 54 held-out)** the reader's forward pass
returned NaN in both passes: the reply is a run of "!", every token log-probability and the hidden state are NaN. They come in
18 whole batches of four, so a numerical overflow in half precision on some images took its batch with it; the same reader had
no such rows on the other three benchmarks. These are failed generations, not answers, and scoring them as wrong would be
false. What I have seen so far: only the two overall accuracies that the scoring script prints (closed-book 64.2%,
with evidence 60.2%, the 72 counted as wrong in both).
**Fix, fixed now:** job J18b reruns exactly these 72 questions, one per batch, in single precision (float32), same prompts and
decoding; their two replies, log-probabilities and hidden states replace the failed rows (the originals are kept in
`experiments/runs/J18/failed_fp16/`). Everything else of v7d and of B2–B4 is unchanged. If a question still fails in float32 it
is excluded from this reader's analysis and counted.

## Outcome log — v7d, CVQA on a second reader (J18 + J18b, analysed 2026-10-10 13:42 IST; preregistered;
## `experiments/results/cvqa_{audit,select}__*__Qwen2.5-VL-3B-Instruct.{json,md}`)
**Data.** J18b reran the 72 failed questions in float32 (version 1 ran out of memory in the evidence pass on one T4; version 2
spread the model over both cards): all 72 finite in both passes and merged (`scripts/merge_rerun.py`; originals in
`experiments/runs/J18/failed_fp16/`). Overall accuracy 64.69% closed-book, 60.62% with evidence.
**R1′ held** (development: helps 10.2%, hurts 15.3%; held-out: 10.2% and 14.0%). **G1′ held:** VGate-full leads the better fixed
policy (never-retrieve) by **+0.85 [+0.24, +1.52]**, again short of one point (main reader: +0.38 [−0.20, +0.96]); it leads the
error gate with the same features by +0.98 [+0.37, +1.60] and told-the-subset by +1.10 [+0.48, +1.73].
**S1′ NOT held, S2′ NOT held.** The after-reading rule leads never-retrieve by **+0.87 [+0.05, +1.73]** (I had required one
point; my estimate was 70%) and VGate-full by +0.02 [−0.66, +0.72]; tokens 1.29 times always-retrieve.
**Why it did not replicate — from the data, variant by variant.** The registered rule has two conditions: the evidence reply
names an option, and it is the more confident one. On the main reader the confidence comparison alone leads by +0.76 [−0.01,
+1.57], and the names-an-option condition adds the rest (+1.62 together), because 3.0% of that reader's evidence replies name
no option and are still the more confident. Qwen2.5-VL-3B never gives such a reply (0.0%), so on it the rule *is* the
confidence comparison: +0.87. The two readers agree on the comparison (+0.76, +0.87); the main reader's one-point lead came
from catching its own unusable replies. So what is tested and what is not: the rule as registered does not carry a one-point
lead to a second reader; a rule built for that reader's failure mode was not tested and is not claimed.
**Planner, B2–B4 (preregistered; held-out, n = 8,373; helps 10.2%, harms 14.0%; needed AUROC 0.643).** B2, "best lead within
0.5 points of the estimate": held for VGate-post (0.82 against 0.56) and the after-reading score (1.91 against 1.51), **not for
VGate-full (1.23 against 0.66)**; "the registered rule earns at most the estimate plus 0.3": held for all three (0.85, 0.76,
0.87 against 0.66, 0.56, 1.51). B3, "leads by a point iff its AUROC exceeds the needed one": correct for VGate-post (0.611, no)
and the after-reading score (0.677, yes), **wrong for VGate-full** (AUROC 0.619 below 0.643, yet its best threshold leads by
1.23). B4, ceiling: held for all three. With B1 and B4 on ReflectiVA's token: rule-level predictions 6 of 6, best-threshold
predictions 2 of 3, lead-by-a-point predictions 2 of 3, ceiling 5 of 6. The estimate again under-predicts a visible lead.

## Notes after claim check 16 (2026-10-10 ~14:30 IST; `docs/experiments/reviewer_simulation_16.md`; no number changes)
- **v8, ReflectiVA: deviations from the release, to be read with the outcome above.** The checkpoint was run with
  transformers 5.16.1 (the release pins an older version) and its first token was read greedily with both token probabilities
  kept (the release samples it at temperature 0.2). The token is [Retrieval] for 99.96% of our held-out E-VQA questions (2,749
  of 2,750); note R13 records 88.4% from their paper for their setting. I cannot say how much of that difference is the
  sampling temperature, the library version or the question set; the audit describes the token as we ran it.
- **v7d result files** are written by the scripts of v7 and v7c and keep their headings: in them "G1: NOT held" means that
  VGate-full does not lead by a point, which is what v7d's G1′ expected (held); "S1 / S2: NOT held" are v7d's S1′ and S2′.
- **v7d, mixed precision.** 72 of the second reader's 10,374 CVQA questions were generated in float32 (J18b), the rest in
  float16. Excluding the 54 held-out ones, or counting them as wrong in both passes, leaves the two registered leads at
  +0.87 / +0.91 and +0.91, never a point (computed by the claim check).
- **v9, forward tests:** the paper now names B3 as well ("the needed AUROC told whether that lead reaches a point" for 2 of 3
  scores) and calls the tests preregistered.
- **v7, fallback analysis:** the paper states the fallback figures again (+0.9, +1.8, +0.0), as the v7 amendment requires;
  under the fallback the one test that held for the main reader (VGate-full against the error gate) does not.
- **v11b's consequence is applied:** the paper says an open substitute judge did not reproduce their judge score.

## Outcome log — v11c (J22, 2026-10-10 12:00–14:30; analysed 2026-10-10 16:34 IST; judge: Qwen2.5-32B-Instruct, 4-bit, greedy;
## `experiments/results/chen_logs_audit__judge32.{json,md}`)
All 6,000 answers judged in one run; no reply failed to start with a number. **Validity gate: NOT passed.** The judge's
never-search and always-search means are 4.4–10.3 points below the printed ones (GPT-4o 41.2 / 44.8 against 51.4 / 52.9; within
5.0 points on both only for no reader, on one of the two for Qwen-VL-2 and DeepSeek-VL-Chat). Closer than the 14B judge
(6.3–13.1), still stricter than Qwen-Max as printed. **As registered: no claim in the paper, and no third open judge.**
The paper says that two open substitute judges did not reproduce their judge score.
**For the record, not a claim:** under this judge too, the hard gate is below told-the-source for every reader (−0.81 [−2.30,
+0.63] for GPT-4o, −0.96, −0.86, −2.24, −1.61 for the others), as under the 14B judge for four of five and under their token
accuracy for all five. Three scorings point the same way; only the token accuracy reproduces what they print.
**What would settle it:** Qwen-Max itself through a DashScope key, or GPT-4o through an OpenAI key (their Table 9), with the
caveat of their README that the judge metric "might be different from the paper".

# PREREG v12 — the planner's noise floor: what a tuned threshold earns from a score with no signal (2026-10-10 16:53 IST; exploratory,
# on data already analysed; written before the quantity is computed for any cell)
**Why.** User, 2026-10-10: the planner's mathematics is classical; can it be made more than that? The ceiling and the estimate
say what a score *with* signal can earn. Missing is the other end: how large a lead the best threshold shows when the score has
*no* signal, because the threshold (and sign) are chosen on the labels they are scored on. That is what separates the "leads
with threshold chosen on the held-out labels" (mean 0.10 over the 156 scores) from the split-half leads (mean −0.02).
**Quantity.** A score independent of the outcome orders the queries at random. With hn helped and mn harmed queries among n,
the cumulative gain along that order is a random path with endpoint (h − m)n, and the best threshold's lead over the better
fixed policy is its maximum minus max(0, endpoint). For a bridge with variance V = 4hmn/(h + m) the expectation is
(√V / 2n) · R(|h − m|n / √V), with R the Mills ratio (1 − Φ)/φ: the **null lead** (one direction of the score). This is the
reflection principle for a Brownian bridge, a textbook result; the claim is only its use as a noise floor for a reported lead.
**Checks (exploratory).** (1) The formula against exact shuffles of the labels (unit test; tolerance 10% or 0.02 points).
(2) For each of the twelve benchmark–reader cells and CVQA: the null lead, one direction and with the sign also chosen (by
shuffling), against the mean best lead of that cell's scores. (3) For the published log of v11 (GPT-4o, 600 questions).
**What I expect.** On E-VQA, InfoSeek, OK-VQA and CVQA the two outcomes are so unbalanced that the null lead is below 0.05
points; on the small, balanced published log it is several tenths of a point. Whatever comes out is reported; the paper gets
one sentence only if it fits without removing a caveat.

## Outcome log — v12, the planner's noise floor (computed 2026-10-10 16:59 IST; `experiments/results/null_lead.{json,md}`; exploratory)
**The formula.** The Brownian-bridge expression of the registration overstates the discrete case by 5–10% on balanced logs and
more on unbalanced ones, so the library uses the exact sum instead (`gate.null_lead`: P(max ≥ b) = C(P+Q, Q+b) / C(P+Q, P) by
reflection); it matches 4,000 shuffles within 6% in the unit test (0.628 against 0.627 points on the published log's counts).
**Case study.** Over the twelve cells the null lead is at most 0.013 points (0.022 with the sign chosen too), as expected for
logs this large and this unbalanced. The 156 scores' best leads average 0.10 points and their split-half leads −0.02; 90 of
156 best leads are above their cell's null lead with sign. So the small tuned leads are mostly not threshold noise; they do
not survive a split because they are small, not because they are selection artefacts.
**Small or balanced logs.** Published log (GPT-4o, 600 questions, helps 7.5%, harms 6.8%): 0.63 points, 0.97 with sign; per
source (100 questions): 0.2–1.5 points, up to 2.4 with sign. Reference log of 1,000 queries with 10% helped and 10% harmed:
0.84 points (1.18 with sign); 10,000 queries: 0.28. A one-point lead of a threshold tuned on the scored labels is within noise
on a balanced log of a thousand queries.
**In the tool and the paper.** `gate.plan` reports it per segment (column `null_lead`); the paper defines it in one sentence
with the reference figure and says it is at most 0.02 points in the case study. No claim of new mathematics: it is the
reflection principle, used as a noise floor.

## Notes on v12 after claim check 17 (2026-10-10 ~18:30 IST; `docs/experiments/reviewer_simulation_17.md`)
- **A reading I withdraw.** "90 of 156 best leads are above their cell's null lead with sign … mostly not threshold noise"
  compared each lead with the *mean* of the null, which a signal-free score itself exceeds in 20–54% of orders (about 60 of 156
  expected). Against the null's 95th percentile, 65 of 156 best leads are above (about 8 expected by chance; the claim check's
  computation). So: more of the tuned leads exceed the noise floor than chance would give, but not "mostly".
- **CVQA** was named in check (2) and is not in `null_lead.json`: the claim check computed 0.018 and 0.031 points in one
  direction and 0.029 and 0.049 with sign for the two readers, below the 0.05 I expected.
- **Wording in the paper:** the null lead is defined as a mean ("the mean lead of a signal-free score's best threshold on the
  labels it is scored on"); on the reference log a signal-free score reaches a point or more in 37% of orders. The case
  study's figure is printed as a ceiling ("below 0.03": the with-sign maximum 0.022, rounded up).
- **Statuses.** The abstract no longer says each result's status is given "where it is reported" (eight places had none of
  the three words); it says results are marked in the text, and three labels were added (the token-saving sentence:
  descriptive; the feedback simulation: registered; the fold-assignment check: exploratory). Unlabelled and to be read as
  descriptive or computed after the result: the CVQA fallback figures, the confidence comparison alone, the comparison with an
  oracle gate, the pilot figures.

# PREREG v13 — a larger reader: Qwen3-VL-32B-Instruct, 4-bit, on a frozen sample of the held-out questions (2026-10-11 19:59 IST; before any
# output of this reader exists)
**Why.** "Open 3–8B models" is the first limitation of the paper. I had told the user that a 32B reader was out of reach on
two T4s (my estimate: 30–50 s per question). That estimate was not measured. J22 (2026-10-10) ran a 32B text model in 4-bit
on the same cards at 1.4 s per short judging prompt, so a 32B vision-language reader at a few seconds per question is
plausible. The user asked on 2026-10-11 what "bigger GPU" meant; this tests whether the answer is "none needed".
**Reader.** Qwen/Qwen3-VL-32B-Instruct, weights in 4-bit NF4 (bitsandbytes), half-precision compute, one copy spread over both
T4s, same prompts, image size, decoding and token budget as the other readers; one question per batch. A quantised reader is
not the full-precision model, and the paper will say "4-bit".
**Questions.** For each of E-VQA, InfoSeek and OK-VQA: the first N questions of a seeded permutation of the held-out split
(`data/splits/<benchmark>_big.txt`, seed 2027, hashes in `SPLITS.sha256`; the job rebuilds the order from the mounted split and
checks the hash). **N is fixed by a rule, not by a result:** a smoke run on 12 questions per benchmark measures seconds per
question; N is the largest of 1,000, 500 and 300 for which both passes on all three benchmarks fit in nine hours of one
session; below 300 the run is not made and this entry is closed as NOT_RUN. The smoke run prints no accuracy.
**Passes and analysis.** Closed-book and top-5 evidence, hidden states of the closed-book pass. `scripts/big_reader.py` →
`experiments/results/big_reader.{json,md}`: per benchmark the two accuracies, the help and harm shares, the planner's floor,
needed AUROC and null lead; on the three benchmarks as mixed traffic, the audit of `gate.audit` with gates fitted out of fold on
this log (the paper's features: uncertainty signals, retriever-score features, 64 components of the hidden state), 95%
image-cluster intervals. There is no development split for this reader, so every gate here is out of fold on held-out data,
the protocol of the paper's log view, not of its Table 2.
**Expectations.** Q1: retrieval helps on net on E-VQA and InfoSeek and hurts on net on OK-VQA (the regimes of the small
readers). Q2: in mixed traffic the benefit gate does not lead told-the-benchmark (point estimate at most zero). Q3: the
benchmark classifier is within one point of the benefit gate. Q4: on no benchmark does the benefit gate, fitted out of fold on
that benchmark, lead the better fixed policy by a point with an interval above zero. My estimates: Q1 80%, Q2 70%, Q3 70%,
Q4 75%. With N = 300 the intervals are about ±5 points per benchmark; the run can show the regimes and the mixed-traffic
pattern, not small within-benchmark leads.
**Consequence for the paper.** Reported whatever comes out, in one sentence of the case study and in the limitations line
("a 32B reader, 4-bit, on N questions per benchmark"). If Q2 or Q3 fail, the paper says that for this reader the gate learned
more than the benchmark.

## v13 — the smoke run and the sample size (2026-10-12 17:29 IST; no accuracy computed)
J23's smoke run (12 questions per benchmark) completed: the 4-bit 32B reader loads on two T4s and answers at 2.8–2.9 s per
question closed-book and 7.2–9.7 s with the evidence (no NaN, no empty reply), 11.3 s for both passes on average. By the
registered rule: N = 1,000 would need 9.4 hours (over the nine allowed), **N = 500 needs about 4.7 hours plus loading: N = 500**
per benchmark, 1,500 questions in all. The full run is pushed with nothing else changed. The smoke run's twelve answers per
benchmark were not scored.

## v13 — the first full run failed (2026-10-13 10:55 IST; nothing scored)
J23 version 2 stopped after 79 minutes: out of memory on the second card at the 292nd E-VQA question of the evidence pass (13 GiB
of weights there left 1.5 GiB for the prompt). Status FAILED; its partial outputs are discarded, not analysed. Version 3, nothing
else changed: less of the weights on the second card (11.5 GiB; two fallback splits if the model then does not fit on the cards),
and the runner retries a question once after clearing memory and otherwise writes no row for it and prints its id. Questions
without both replies are excluded from this reader's analysis and counted in the outcome.

## Outcome log — v13, a larger reader (J23 version 3, 11:00–16:07; analysed 2026-10-13 16:12 IST; preregistered;
## `experiments/results/big_reader.{json,md}`, `big_reader__log.csv`)
**Run.** The two tighter memory splits did not fit the model on the cards; the third (10 and 13 GiB of weights, as in the
smoke run) ran all three benchmarks with the out-of-memory guard: **2 of 1,500 questions** (EVQA_2515, EVQA_3534, evidence
pass) got no reply after a retry and are excluded; 1,498 questions are analysed. No NaN, no empty reply.
**All four expectations held.** Q1: closed-book / with evidence 24.5 / 49.8 on E-VQA, 12.0 / 22.8 on InfoSeek, 62.4 / 32.6 on
OK-VQA: the same three regimes as the small readers (helps 31.7 / 14.4 / 3.0%, harms 6.4 / 3.6 / 32.8%). Q2: in mixed traffic
the benefit gate (fitted out of fold on this log) reaches 43.5 against 45.0 for told-the-benchmark: **−1.5 [−2.4, −0.5]**. Q3: the
benchmark classifier reaches 44.3 (−0.7 [−1.2, −0.3] against told), 0.8 above the gate. Q4: within a benchmark the gate's lead
over the better fixed policy is −3.61 [−5.48, −1.88], −1.00 [−2.22, +0.00] and −1.60 [−3.00, −0.20]; its help-vs-harm AUROC
(0.63, 0.63, 0.75) is below the needed one (0.83, 0.87, 0.94) on each. Always-retrieve 35.1, never 33.0, oracle gate 49.3.
**Reading and limits.** The gate gains 8.4 points over always-retrieve and a classifier of the benchmark gains 9.2: on a
reader four times the size of our largest, the gate's gain is again the benchmark's. Limits: 4-bit weights, not the
full-precision model; 500 questions per benchmark; gates fitted out of fold on the held-out log (there is no development
split for this reader), which with 500 questions per benchmark is a small fit and may understate what a gate could learn
from more data; one run. The closed-book accuracy of this quantised reader is not above the 4B reader's on InfoSeek (12.0 on
this sample against 17.8 on the full split), which I did not expect and have not explained.

## Notes on v13 after claim check 19 (2026-10-14 ~19:30 IST; `docs/experiments/reviewer_simulation_19.md`)
- **The 32B reader is not the most accurate one.** On the same 1,498 questions (closed-book / with evidence): E-VQA 24.5 / 49.8
  (8B 26.9 / 52.6; 4B 23.9 / 49.6), InfoSeek 12.0 / 22.8 (8B 16.8 / 29.6; 4B 14.8 / 25.6), OK-VQA 62.4 / 32.6 (8B 62.4 / 45.4; 4B
  60.4 / 40.0). With 4-bit weights it is below the 8B reader in both passes on every benchmark (one tie) and below the 4B
  reader with evidence on InfoSeek and OK-VQA. So it tests the pattern on a larger, not a better, reader; the paper now says
  "a 4-bit 32B reader (four times our largest, but less accurate here than our 8B one)". Why 4-bit weights cost this much here
  is not established.
- J23's database rows had the wrong build stamp; the run log shows `1ce75901f019` (corrected).

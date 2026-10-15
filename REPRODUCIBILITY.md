# Reproducing VGate's results

Every number in the paper is generated: `paper/tables/numbers.tex` (in-text), `paper/tables/main.tex` (Table 1) and
`paper/figures/frontier.pdf` (Fig. 2) come from the run outputs via `scripts/paper_numbers.py` and
`scripts/generate_results.py`. Run-level provenance (git SHA, build stamp, split hash, platform, cost) is in
`experiments/results/experiment_database.csv`. Criteria were preregistered in `experiments/PREREG.md` (v0 at git
2e8c41c, before any scoring; v1 at e871e87, before any held-out output; v2 / v2b at 1db7dc3 / 797c2b2, before any
OK-VQA output; v2c at 3f99d15, before any Qwen3-VL-8B output; v2d at 1631fbe; v2e, exploratory, at 6199e65). The
outcome log at the end of that file records every result, including failed runs and unsupported hypotheses.

## 1. Environment
- Local, CPU (data, scoring, analysis, demo, paper): Python 3.13, `make setup` (torch CPU, transformers 5.17,
  scikit-learn 1.9, pandas 3.0). LaTeX via `tectonic`.
- GPU inference: Kaggle T4×2 (driver 580, torch 2.10+cu128) or Colab T4, with **transformers 5.16.1** pinned, fp16,
  `attn_implementation="sdpa"`. The A2000 (Ampere) can run the same code in bf16.

## 2. Data (`data/README.md`)
1. Download (about 1.3 GB): HF `BByrneLab/EVQA-test-with-PreFLMR-Lretrieval`,
   `BByrneLab/Infoseek-test-with-EvaClip-retrieval`, `BByrneLab/Infoseek-EchoSight-100K-KB`,
   `BByrneLab/BERAG-demo-InfoSeek-images`, M2KR `EVQA_data`/`EVQA_passages`/`Infoseek_data` test parquets, and the
   E-VQA inference images from the ReflectiVA release (`evqa_inference_images.zip`).
2. `make data`, which writes `data/processed/{evqa,infoseek}.jsonl` and the frozen splits `data/splits/*.txt`.
   Check `sha256sum data/splits/*.txt` against `data/splits/SPLITS.sha256`.
3. OK-VQA: M2KR `OKVQA_data` / `OKVQA_passages` test parquets and the COCO val2014 images of its 5,046 questions
   (`data/raw/okvqa_images/val2014`). Retrieval is `kaggle/okvqa-retrieval` (bge-base-en-v1.5; the same code as
   `scripts/build_okvqa_retrieval.py`); its output `data/raw/okvqa_retrieval.parquet` has sha256 `d2c574d7…c41cd4`.
   Then `PYTHONPATH=src python -m vgate.data okvqa`.

## 3. Inference (GPU)
Each job is a folder `kaggle/<job>/` (`kernel-metadata.json` + `run.py`). Push with `scripts/push_kernel.sh kaggle/<job>`;
this syncs `src/` as a Kaggle dataset and bakes its build stamp into the job. Wait with
`scripts/wait_kaggle.sh <job>`; fetch with `scripts/kaggle.sh kernels output <user>/<job> -p experiments/runs/<X>`.

| run | job | what | approx. T4 time |
|---|---|---|---|
| J1 | `g1-c1-q3vl4b` / `g1-c1-resume` | Qwen3-VL-4B, dev, all conditions + 100-q rerun | 2.6 h |
| J2 | `g1-repro-q25vl3b` | Qwen2.5-VL-3B, full sets (ne/ev/oracle) + dev features | 3 h |
| J3 | `c2-think` | Qwen3-VL-4B-Thinking pilot (dev) | 7.5 h |
| J4 | `j4-confirm-q3vl4b` | Qwen3-VL-4B, held-out (ne/ev/oracle) | 1.8 h |
| J5 | `j5-internvl` (+ CPU `j5-bundle`) | InternVL3.5-4B-HF, all questions (ne/ev), single-tile images | 2 h |
| J6 | `j6-okvqa` | OK-VQA, Qwen3-VL-4B + Qwen2.5-VL-3B, all 5,046 questions (ne/ev) | 1.5 h |
| J7 | `j7-q3vl8b` | Qwen3-VL-8B (split over both T4s), E-VQA + InfoSeek (ne/ev) | see outcome log |
| J8 | `j8-evhidden` | evidence-pass hidden states, Qwen3-VL-4B, three datasets | see outcome log |

A single-machine equivalent: `PYTHONPATH=src python -m vgate.run --data data --images data/raw --dataset evqa
--split confirm --model Qwen/Qwen3-VL-4B-Instruct --conditions ne,ev,oracle --out experiments/runs/J4`.

## 4. Scoring, analysis, paper
```
make score      # official metrics (E-VQA BEM runs on CPU and takes minutes)
make confirm    # preregistered held-out analysis -> experiments/runs/<run>/confirm__*.{json,md}
make extend     # VGate-full, four-outcome probe, answer selection, H-OK -> experiments/runs/<run>/extend__*.{json,md}
make results    # operating points (mixed traffic), Tables 1-2, numbers.tex
make screenshot # Fig. 2 (demo Inspector; needs playwright + chromium)
make paper      # paper/main.pdf (4 pages)
make test       # 59 unit tests (metrics, labels, leakage whitelist, gate math, audit, planner)
```
Expected primary endpoint (Qwen3-VL-4B, held-out): dnAUC(benefit_post − error_post) = +1.64 pt, 95% CI
[+1.26, +2.01]. The seeds are fixed (splits 2027, bootstrap 0), so the numbers reproduce exactly from the same run
outputs.

## 5. Demo
`make demo` starts the cached Gradio app on the held-out outputs (http://localhost:7860). It needs
`experiments/runs/{J4,J6}/gate_scores__*.csv`, which `make confirm` writes and `make extend` completes.

## Verification log
- 2026-10-03: from a fresh clone of `research` with only the raw model outputs copied in (`experiments/runs/J1–J4`
  JSONL, hidden states, meta) and the data symlinked, the following produced byte-identical
  `paper/tables/main.tex`, `paper/tables/numbers.tex` and `experiments/results/signal_auroc_confirm.csv`, and a
  4-page PDF:
  1. split hashes verified;
  2. `make test` (42 passed);
  3. `make score` (all scores recomputed);
  4. `make confirm`;
  5. `make results`;
  6. `make paper`.
- 2026-10-04 (19:15–19:48 IST, commit `00f8034`): fresh-clone re-verification of the extended pipeline. From
  `git clone` with only the raw model outputs linked in (`experiments/runs/J1–J7`: generations, hidden states, run
  metadata) and `data/raw`, `data/processed` linked:
  1. split hashes verified (six files);
  2. `make test` (44 passed);
  3. `make score` (10 min), `make confirm` (11 min), `make extend` (5 min), `make results` (7 min), `make paper`;
  4. **byte-identical** to the working tree: `paper/tables/{main,numbers,mixed,replication,primary}.tex`,
     `experiments/results/{signal_auroc_confirm,pass_costs,label_efficiency,multiplicity}.csv` and the three
     `operating_point__*.json` files; the PDF has 4 pages.
  Not covered by this check: the J8 selector analysis and the J9 live-pipeline check (not finished at the time), the
  frozen gates (`scripts/freeze_gates.py`, which self-checks against `operating_point__*.json`), and Fig. 2
  (`make screenshot`, which needs a browser).
- 2026-10-04 (21:04–21:37 IST, commit `f0b669e`): the same fresh-clone check repeated after the final paper edits
  (claim check 4, Tables 2–3 changed, live-pipeline numbers added). Result: the same five paper tables and seven
  result files are byte-identical to the working tree; 44 tests pass; the PDF has 4 pages. The live-pipeline numbers
  in `numbers.tex` are read from the tracked `experiments/results/live_check.json`; regenerating that file needs the
  J9 outputs (`make live`), which this check did not re-run.
- 2026-10-05 (03:14–03:53 IST, commit `2fe5555`): the same fresh-clone check after the loss decomposition (PREREG v3a)
  and the one-sided-feedback result (v3b) entered the paper and `make results` gained `scripts/loss_decomposition.py`.
  Result: 44 tests pass; the five paper tables, the seven result files above and
  `experiments/results/loss_decomposition__confirm.{json,md}` are byte-identical to the working tree; the PDF has 4
  pages. As before, `make live` was not re-run: `online_gate__confirm.json` (v3b) and `live_check.json` are read from
  the tracked files; regenerating them needs the J9 outputs and the frozen gates (`make gates`, then `make live`).
- 2026-10-05 (04:26–05:05 IST, commit `7e761dd`): repeated after the claim-check fixes (like-for-like gap added to
  `loss_decomposition.py`, paper text and macros changed, second-model feedback simulation). Same result: 44 tests
  pass, the same 14 files are byte-identical to the working tree, the PDF has 4 pages.
- 2026-10-06 (01:10–02:03 IST, commit `0ffb730`): the check, extended to the two models added on 2026-10-05 (raw outputs of
  J11 and J12 linked; `make results` now also runs `operating_point.py` for four models and `lodo.py` twice). All 18 compared
  result and table files were byte-identical **except `paper/tables/numbers.tex`**, and `make paper` failed: six macros of
  the hidden-state selector (PREREG v2d) were missing because `make extend` did not pass `--ev_hidden_run experiments/runs/J8`
  (the tracked results had been produced with that argument by hand). Fixed in the Makefile (`3e867ad`).
- 2026-10-06 (02:04–02:57 IST, commit `3e867ad`): repeated with J8's raw outputs linked as well. 45 tests pass; `make score`,
  `confirm`, `extend`, `results`, `paper` all succeed; the five paper tables (including `numbers.tex`), the seven earlier
  result files, `loss_decomposition__confirm.{json,md}`, `operating_point__*.json` for all four models and
  `lodo__confirm{,__v4c}.json` are byte-identical to the working tree; the PDF has 4 pages. Not regenerated by this check
  (read from tracked files): `live_check.json` and `online_gate__confirm*.json` (`make gates live`, needs J9), and the
  development-data pilots (`make pilots`, needs J10 and J13).
- 2026-10-07 (00:48–02:28 IST, commit `d4d5da4`, and 02:28–03:41 IST, commit `1ead135`): the check with the raw outputs of J1–J8
  and J11–J16 linked and `make test score confirm extend external cvqa results paper`. Every target succeeds; 59 tests pass; the
  PDF has 4 pages. Of 38 compared files (the earlier ones, the published-gate and CVQA results, `tables/plan.tex`,
  `gate_bound.{json,md}`, `tail_gate_check.json`, the five `tail_gate_diag*.json`, `tail_gate_dev.json`, `demo/sample_log.csv`)
  35 were byte-identical to the working tree. **Three were not: `gate_bound.json`, `gate_bound.md` and `paper/tables/numbers.tex`.**
  Cause: in the working tree the intermediate files `experiments/runs/J2/gate_scores__{evqa,infoseek}__…__confirm.csv` and the
  same two of J5 were older than `make extend` (they lacked the `vgate_full` and `probe4` scores), so `scripts/gate_bound.py` had
  counted 148 scores where a run from the raw outputs gives 156. `make extend` was re-run in the working tree and the three files
  regenerated (2026-10-07 03:52): all 38 are now byte-identical to the fresh clone. What changed in the paper: 148 → 156 scores,
  80 → 86 above the estimate, split-half mean −0.01 → -0.02; every other quoted number of that paragraph is the
  same at its printed precision.
  Not regenerated by this check (read from tracked result files): `live_check.json`, `online_gate__confirm*.json` (need J9), the
  development-data pilots of `make pilots` (J10, J13), and the image re-ranking results `kbimage_rerank*.json` and
  `kbimage_reader_pass.json` (they need the candidate pages' lead images, which are fetched from Wikipedia and not
  redistributed, and the raw outputs of J19).

- 2026-10-07 (afternoon) — results added after that check, and how each is regenerated:
  - `published_rates.{json,md}` (`scripts/published_rates.py`, part of `make results`): no data needed; the counts are copied
    from arXiv:2405.18740v1, Tables 1, 3 and 4, and are in the script.
  - `chen_logs_audit.{json,md}` and `demo/sample_log_published.csv` (`scripts/chen_logs_audit.py`, part of `make results`): read
    the tracked table `experiments/results/chen_logs__questions.csv`. That table is produced by `scripts/chen_logs_score.py`
    from a copy of github.com/Chord-Chen-30/VLLM-KnowledgeBoundary (Apache-2.0; the Git-LFS files of `evaluation/eval_data/` are
    fetched from `media.githubusercontent.com`) with the authors' own accuracy function; it needs `jieba` and `nltk` (resource
    `punkt_tab`), which are not dependencies of this repository (`pip install --target <folder> jieba nltk`, then `--deps
    <folder>`). Their LLM-judge score is not computed. Check: the Mix accuracies of their Tables 3 and 11 are reproduced exactly
    for four readers and to 0.14 points for Qwen-VL-2.
  - `rerank_gate_pilot.{json,md}`, `kbimage_reader_pass__questions.csv` (and the `__confirm` versions): need the candidate
    pages' lead images and the raw outputs of J19 / J20, like the other `kbimage_*` results; not part of `make results`.
- 2026-10-10 — runs added and how they enter the build: **J17** (ReflectiVA's retrieval token; transformers 5.16.1 instead of the
  release's pinned version, greedy first token instead of sampling at 0.2) is read by `make external`; **J18** (Qwen2.5-VL-3B on
  CVQA) by the new target `make cvqa2`, which also runs the planner's forward tests (`gate_bound.py --prospective`). In J18,
  72 questions returned NaN in float16 and were rerun in float32 by **J18b** and merged with `scripts/merge_rerun.py` (failed
  originals in `experiments/runs/J18/failed_fp16/`); `experiments/runs/J18` as it stands is the merged run. **J21** (an open
  judge for Chen et al.'s answers) failed its registered validity gate and feeds no number of the paper
  (`chen_logs_audit.py --metric judge` is not part of `make results`).
- 2026-10-10 (evening): `null_lead.{json,md}` (`scripts/null_lead_check.py`, part of `make results`; PREREG v12) is computed
  from `gate_bound.json` and `demo/sample_log_published.csv`; its shuffles are seeded.
- 2026-10-10 (17:27–19:27 IST, commit `f61a0e8`): **final check on the submission state.** A fresh clone with the raw outputs
  of J1–J8 and J11–J18 linked (J18 as merged with J18b) and `make test score confirm extend external cvqa cvqa2 results paper`:
  every target succeeds; 60 tests pass; the PDF has 4 pages; **all 44 compared files are byte-identical to the working tree**
  (the 38 of the earlier check, the second reader's CVQA audit and after-reading results, `gate_bound__prospective.json`,
  `published_rates.json`, `chen_logs_audit.json`, `null_lead.json`, `demo/sample_log_published.csv`). An earlier run on
  `9477096` (13:46–15:16) had 43 of 43 identical. Not regenerated by this check and read from tracked files:
  `live_check.json`, `online_gate__confirm*.json` (J9), the development pilots of `make pilots` (J10, J13), the image
  re-ranking results (`kbimage_*`, `rerank_gate_pilot*`; they need the candidate pages' lead images and J19 / J20),
  `chen_logs__questions.csv` (needs a copy of the other group's repository, `jieba` and `nltk`), and the two judge analyses
  (J21, J22), which feed no number of the paper.
- 2026-10-14 (20:50–22:30 IST, commit `c7a1d18`): **final check on the submission state** with `scripts/fresh_clone_check.sh`
  (now part of the repository): a fresh clone with the raw outputs of J1–J8, J11–J18 and J23 linked and `make test score confirm
  extend external cvqa cvqa2 bigreader results paper`. All nine split hashes match; every target succeeds; the PDF has 4 pages;
  **all 46 compared files are byte-identical to the working tree** (the 44 of the 2026-10-10 check plus `big_reader.json` and
  `big_reader__log.csv`).
- 2026-10-15 (19:27–21:20 IST, commit `9713e49`): the same check after PREREG v14 / v14b (`make results` now also runs
  `chen_logs_audit.py --sources ... --tag bing` and `strong_gate.py --phase forecast` then `--phase test`). Every target succeeds;
  the PDF has 4 pages; **all 49 compared files are byte-identical** (the 46 above plus `chen_logs_audit__bing.json`,
  `strong_gate__forecast.json` and `strong_gate.json`).

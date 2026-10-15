# VGate — common targets. Heavy GPU inference runs on Kaggle (kaggle/<job>), see REPRODUCIBILITY.md.
PY := .venv/bin/python
M1 := Qwen3-VL-4B-Instruct
M2 := Qwen2.5-VL-3B-Instruct

M3 := InternVL3_5-4B-HF
M4 := Qwen3-VL-8B-Instruct

.PHONY: setup data test score confirm extend results gates live pilots screenshot paper demo all
setup:        ## local env (CPU): data prep, scoring, analysis, demo
	python3 -m venv .venv && .venv/bin/pip install -e ".[model,demo,dev]" rank_bm25 tabulate pypdf
data:         ## unified tables + frozen image-grouped splits (needs data/raw, see data/README.md)
	PYTHONPATH=src $(PY) -m vgate.data
test:
	$(PY) -m pytest -q tests/
score:        ## official metrics for every run
	for d in evqa infoseek; do $(PY) scripts/score.py --run experiments/runs/J4 --dataset $$d --model $(M1); \
	  $(PY) scripts/score.py --run experiments/runs/J1 --dataset $$d --model $(M1); \
	  $(PY) scripts/score.py --run experiments/runs/J2 --dataset $$d --model $(M2); \
	  $(PY) scripts/score.py --run experiments/runs/J3 --dataset $$d --model Qwen3-VL-4B-Thinking; \
	  $(PY) scripts/score.py --run experiments/runs/J3 --dataset $$d --model $(M1); \
	  $(PY) scripts/score.py --run experiments/runs/J5 --dataset $$d --model $(M3); \
	  $(PY) scripts/score.py --run experiments/runs/J7 --dataset $$d --model $(M4); done
	for m in $(M1) $(M2); do $(PY) scripts/score.py --run experiments/runs/J6 --dataset okvqa --model $$m; done
	$(PY) scripts/score.py --run experiments/runs/J11 --dataset okvqa --model $(M3)
	$(PY) scripts/score.py --run experiments/runs/J12 --dataset okvqa --model $(M4)
confirm:      ## preregistered held-out analysis (primary endpoint + secondaries)
	$(PY) scripts/confirm.py --train_run experiments/runs/J1 --test_run experiments/runs/J4 --model $(M1) --transfer
	$(PY) scripts/confirm.py --train_run experiments/runs/J1 --test_run experiments/runs/J4 --model $(M1) --labels strict
	$(PY) scripts/confirm.py --train_run experiments/runs/J2 --test_run experiments/runs/J2 --model $(M2) --transfer
	$(PY) scripts/confirm.py --train_run experiments/runs/J5 --test_run experiments/runs/J5 --model $(M3)
	$(PY) scripts/confirm.py --train_run experiments/runs/J7 --test_run experiments/runs/J7 --model $(M4)
	for m in $(M1) $(M2); do $(PY) scripts/confirm.py --train_run experiments/runs/J6 --test_run experiments/runs/J6 --model $$m --datasets okvqa; done
	$(PY) scripts/confirm.py --train_run experiments/runs/J11 --test_run experiments/runs/J11 --model $(M3) --datasets okvqa
	$(PY) scripts/confirm.py --train_run experiments/runs/J12 --test_run experiments/runs/J12 --model $(M4) --datasets okvqa
extend:       ## PREREG v2/v2b/v2c/v2d: VGate-full, four-outcome probe, VGate-select (v2d: with evidence-pass hidden states, J8), H-OK (run after confirm)
	$(PY) scripts/extend.py --train_run experiments/runs/J1 --test_run experiments/runs/J4 --model $(M1) --ev_hidden_run experiments/runs/J8
	$(PY) scripts/extend.py --train_run experiments/runs/J2 --test_run experiments/runs/J2 --model $(M2)
	$(PY) scripts/extend.py --train_run experiments/runs/J5 --test_run experiments/runs/J5 --model $(M3)
	$(PY) scripts/extend.py --train_run experiments/runs/J7 --test_run experiments/runs/J7 --model $(M4)
	$(PY) scripts/extend.py --train_run experiments/runs/J6 --test_run experiments/runs/J6 --model $(M1) --datasets okvqa --ev_hidden_run experiments/runs/J8
	$(PY) scripts/extend.py --train_run experiments/runs/J6 --test_run experiments/runs/J6 --model $(M2) --datasets okvqa
	$(PY) scripts/extend.py --train_run experiments/runs/J11 --test_run experiments/runs/J11 --model $(M3) --datasets okvqa
	$(PY) scripts/extend.py --train_run experiments/runs/J12 --test_run experiments/runs/J12 --model $(M4) --datasets okvqa
results:      ## Tables 1-2, the frontier figure and every in-text number (paper/tables/numbers.tex)
	$(PY) scripts/extra_analyses.py
	$(PY) scripts/operating_point.py --model $(M1) --runs evqa=J1:J4,infoseek=J1:J4,okvqa=J6:J6
	$(PY) scripts/operating_point.py --model $(M1) --runs evqa=J1:J4,infoseek=J1:J4,okvqa=J6:J6 --labels relaxed
	$(PY) scripts/operating_point.py --model $(M2) --runs evqa=J2:J2,infoseek=J2:J2,okvqa=J6:J6
	$(PY) scripts/operating_point.py --model $(M3) --runs evqa=J5:J5,infoseek=J5:J5,okvqa=J11:J11
	$(PY) scripts/operating_point.py --model $(M4) --runs evqa=J7:J7,infoseek=J7:J7,okvqa=J12:J12
	$(PY) scripts/lodo.py --split confirm
	$(PY) scripts/lodo.py --split confirm --models $(M3),$(M4) --tag v4c
	$(PY) scripts/multiplicity.py
	$(PY) scripts/loss_decomposition.py --split confirm
	$(PY) scripts/generate_results.py experiments/runs/J4/confirm__$(M1).json --extra experiments/runs/J6/confirm__$(M1).json --out paper
	$(PY) scripts/make_sample_log.py
	$(PY) scripts/gate_bound.py
	$(PY) scripts/tail_gate_check.py
	$(PY) scripts/tail_gate_diag.py
	$(PY) scripts/tail_gate_diag.py --model $(M2)
	$(PY) scripts/tail_gate_diag.py --model $(M3)
	$(PY) scripts/tail_gate_diag.py --model $(M4)
	$(PY) scripts/tail_gate_diag.py --cvqa
	$(PY) scripts/tail_gate_dev.py
	$(PY) scripts/published_rates.py
	$(PY) scripts/chen_logs_audit.py
	$(PY) scripts/chen_logs_audit.py --sources "life,in-house 2,nocaps,visual7w" --tag bing
	$(PY) scripts/strong_gate.py --phase forecast
	$(PY) scripts/strong_gate.py --phase test
	$(PY) scripts/null_lead_check.py
	$(PY) scripts/paper_numbers.py
gates:        ## frozen gate arrays for the live pipeline (self-checks against the operating-point results)
	$(PY) scripts/freeze_gates.py --model $(M1)
	$(PY) scripts/freeze_gates.py --model $(M2) --runs evqa=J2:J2,infoseek=J2:J2,okvqa=J6:J6
	$(PY) scripts/freeze_gates.py --model $(M3) --runs evqa=J5:J5,infoseek=J5:J5,okvqa=J11:J11
	$(PY) scripts/freeze_gates.py --model $(M4) --runs evqa=J7:J7,infoseek=J7:J7,okvqa=J12:J12
live:         ## live-pipeline check, refit on live logs, and the one-sided-feedback simulation (needs experiments/runs/J9; run after gates)
	$(PY) scripts/live_report.py
	$(PY) scripts/online_gate.py --split confirm
	$(PY) scripts/online_gate.py --split confirm --model $(M2)
	$(PY) scripts/online_gate.py --split confirm --model $(M3)
	$(PY) scripts/online_gate.py --split confirm --model $(M4)
pilots:       ## development-data pilots (PREREG v3, v5); need experiments/runs/J10 and J13 (score J10 first: scripts/score.py --run experiments/runs/J10 ...)
	$(PY) scripts/passage_select.py --run experiments/runs/J10 --logged evqa=J1,infoseek=J1,okvqa=J6 --split dev
	$(PY) scripts/pilot_explore.py --run experiments/runs/J10 --logged evqa=J1,infoseek=J1,okvqa=J6
	$(PY) scripts/passage_select.py --run experiments/runs/J13 --logged evqa=J1,infoseek=J1,okvqa=J6 --split dev
	$(PY) scripts/critic_pilot.py --run experiments/runs/J13 --logged evqa=J1,infoseek=J1,okvqa=J6
	$(PY) scripts/critic_pilot.py --run experiments/runs/J10 --logged evqa=J1,infoseek=J1,okvqa=J6 --tag reader_check
external:     ## PREREG v8: published gates through the audit (needs experiments/runs/J13 and J16; J17 when it exists); run before results
	$(PY) scripts/external_gates.py --split confirm
	$(PY) scripts/external_gates.py --split dev
cvqa:         ## PREREG v7: the fourth benchmark (needs experiments/runs/J14 with its data/ folder, and J15); run before results
	$(PY) scripts/score.py --run experiments/runs/J14 --dataset cvqa --model $(M1) --data experiments/runs/J14/data
	$(PY) scripts/score.py --run experiments/runs/J15 --dataset cvqa --model $(M1) --data experiments/runs/J14/data
	$(PY) scripts/cvqa_audit.py --split dev
	$(PY) scripts/cvqa_audit.py --split dev --fallback
	$(PY) scripts/cvqa_audit.py --split confirm
	$(PY) scripts/cvqa_audit.py --split confirm --fallback
	$(PY) scripts/cvqa_select.py --split dev
	$(PY) scripts/cvqa_select.py --split confirm
	$(PY) scripts/cvqa_select.py --split confirm --fallback
	$(PY) scripts/select_open_ended_dev.py
cvqa2:        ## PREREG v7d and v9 (B1-B4): the second reader on CVQA (needs experiments/runs/J18, J14/data and J17); run before results
	$(PY) scripts/score.py --run experiments/runs/J18 --dataset cvqa --model $(M2) --data experiments/runs/J14/data
	for s in dev confirm; do for f in "" "--fallback"; do $(PY) scripts/cvqa_audit.py --split $$s $$f --model $(M2) --run_dev experiments/runs/J18 --run_confirm experiments/runs/J18 || exit 1; done; done
	$(PY) scripts/cvqa_select.py --split dev --model $(M2) --run_dev experiments/runs/J18 --run_confirm experiments/runs/J18
	$(PY) scripts/cvqa_select.py --split confirm --model $(M2) --run_dev experiments/runs/J18 --run_confirm experiments/runs/J18
	$(PY) scripts/cvqa_select.py --split confirm --fallback --model $(M2) --run_dev experiments/runs/J18 --run_confirm experiments/runs/J18
	$(PY) scripts/gate_bound.py --prospective --b1_gate reflectiva --cvqa_model $(M2) --cvqa_run_dev experiments/runs/J18 --cvqa_run_confirm experiments/runs/J18
bigreader:    ## PREREG v13: the 32B reader in 4-bit on a frozen sample (needs experiments/runs/J23); run before results
	for d in evqa infoseek okvqa; do $(PY) scripts/score.py --run experiments/runs/J23 --dataset $$d --model Qwen3-VL-32B-Instruct || exit 1; done
	$(PY) scripts/big_reader.py
screenshot:   ## Figs. 2-3 (needs playwright + chromium); launches the demo and captures the Inspector and the audit table
	$(PY) scripts/demo_screenshot.py
	$(PY) scripts/demo_screenshot_audit.py --log_only --published --width 640 --out paper/figures/demo_audit_published.png
	$(PY) scripts/demo_screenshot_audit.py
paper:
	cd paper && tectonic -X compile main.tex
demo:
	$(PY) demo/app.py --run experiments/runs/J4,experiments/runs/J6 --model $(M1) --split confirm
all: test score confirm extend results paper

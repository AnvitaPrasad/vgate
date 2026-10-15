#!/usr/bin/env bash
# Fresh-clone reproducibility check: clone the repository into build/clone, bring in only the raw model outputs and the data,
# regenerate everything with make, and compare the result files and the paper's tables with the working tree.
#   caffeinate -i bash scripts/fresh_clone_check.sh > build/fresh_clone.log 2>&1     (about two hours on a laptop)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="$REPO/build/clone"; rm -rf "$C"; git clone -q "$REPO" "$C"; cd "$C" || exit 1
echo "clone at $(git rev-parse --short HEAD)"
ln -s "$REPO/.venv" .venv; ln -s "$REPO/data/raw" data/raw; ln -s "$REPO/data/processed" data/processed
mkdir -p experiments/runs
for J in J1 J2 J3 J4 J5 J6 J7 J8 J11 J12 J13 J14 J15 J16 J17 J18 J23; do
  mkdir -p experiments/runs/$J
  # raw model outputs only: generations, hidden states, run metadata (no scores, analyses or gate scores)
  find "$REPO/experiments/runs/$J" -maxdepth 1 -type f \( -name "*__ne.jsonl" -o -name "*__ev.jsonl" -o -name "*__oracle.jsonl" \
     -o -name "*__ne_sample.jsonl" -o -name "*__ent.jsonl" -o -name "*__ent_sample.jsonl" -o -name "*__ptrue.jsonl" -o -name "*__para.jsonl" \
     -o -name "*__think*.jsonl" -o -name "*__ne_tset.jsonl" -o -name "*_rerun.jsonl" -o -name "hidden_*" -o -name "hidden-ev_*" -o -name "meta__*.json" \
     -o -name "*__pv_[0-9].jsonl" -o -name "*__gate_*.jsonl" -o -name "*__gate.jsonl" \) \
     -exec ln -s {} experiments/runs/$J/ \;
  echo "$J: $(ls experiments/runs/$J | wc -l | tr -d ' ') raw files"
done
# the fourth benchmark's question records and splits were written by its GPU job (no images)
ln -s "$REPO/experiments/runs/J14/data" experiments/runs/J14/data
export PATH=/opt/homebrew/bin:$PATH
( cd data/splits && shasum -a 256 -c SPLITS.sha256 2>&1 | tail -9 )
for t in test score confirm extend external cvqa cvqa2 bigreader results paper; do
  echo "=== make $t ($(date +%H:%M))"; make $t > "$REPO/build/clone_make_$t.log" 2>&1; echo "rc=$? ($(date +%H:%M))"; tail -2 "$REPO/build/clone_make_$t.log" | cut -c1-200
done
echo "=== comparison with the working tree"
for f in paper/tables/main.tex paper/tables/numbers.tex paper/tables/mixed.tex paper/tables/replication.tex paper/tables/primary.tex paper/tables/plan.tex \
         experiments/results/signal_auroc_confirm.csv experiments/results/pass_costs.csv experiments/results/label_efficiency.csv \
         experiments/results/multiplicity.csv experiments/results/operating_point__Qwen3-VL-4B-Instruct.json \
         experiments/results/operating_point__Qwen2.5-VL-3B-Instruct.json experiments/results/operating_point__Qwen3-VL-4B-Instruct__relaxed.json \
         experiments/results/loss_decomposition__confirm.json experiments/results/loss_decomposition__confirm.md \
         experiments/results/operating_point__InternVL3_5-4B-HF.json experiments/results/operating_point__Qwen3-VL-8B-Instruct.json \
         experiments/results/lodo__confirm.json experiments/results/lodo__confirm__v4c.json \
         experiments/results/external_gates__confirm.json experiments/results/external_gates__dev.json \
         experiments/results/cvqa_audit__dev.json experiments/results/cvqa_audit__confirm.json \
         experiments/results/cvqa_audit__confirm__fallback.json experiments/results/cvqa_select__confirm.json experiments/results/cvqa_select__confirm__fallback.json \
         experiments/results/select_open_ended__dev.json \
         experiments/results/gate_bound.json experiments/results/gate_bound.md \
         experiments/results/tail_gate_check.json experiments/results/tail_gate_diag.json experiments/results/tail_gate_diag__cvqa.json \
         experiments/results/tail_gate_diag__Qwen2.5-VL-3B-Instruct.json experiments/results/tail_gate_diag__InternVL3_5-4B-HF.json \
         experiments/results/tail_gate_diag__Qwen3-VL-8B-Instruct.json experiments/results/tail_gate_dev.json \
         experiments/results/cvqa_audit__confirm__Qwen2.5-VL-3B-Instruct.json experiments/results/cvqa_select__confirm__Qwen2.5-VL-3B-Instruct.json \
         experiments/results/gate_bound__prospective.json experiments/results/published_rates.json experiments/results/chen_logs_audit.json \
         experiments/results/null_lead.json experiments/results/big_reader.json experiments/results/big_reader__log.csv \
         experiments/results/chen_logs_audit__bing.json experiments/results/strong_gate__forecast.json experiments/results/strong_gate.json \
         demo/sample_log_published.csv demo/sample_log.csv; do
  if cmp -s "$C/$f" "$REPO/$f"; then echo "IDENTICAL  $f"; else echo "DIFFERENT  $f"; fi
done
"$REPO/.venv/bin/python" -c "import pypdf; print('pages:', len(pypdf.PdfReader('paper/main.pdf').pages))"

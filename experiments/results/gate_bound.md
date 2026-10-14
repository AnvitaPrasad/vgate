# What a read-or-skip gate can earn: a ceiling and a three-number estimate against every held-out score (exploratory)

156 scores (benchmark x reader x gate or signal). Observed = best lead over the better fixed policy that the score reaches with threshold and sign chosen on the held-out labels; split = both chosen on one half of the images and applied to the other. Ceiling = min(h, m)(2A - 1), a bound for any concave ROC curve; estimate = G*(h, m, A), an equal-variance binormal ROC curve.

**Estimate.** Mean absolute difference to the observed lead 0.08 points (largest 0.71; mean signed difference +0.08; 98.1% within half a point); correlation 0.801. It is not a bound: 86 of 156 observed leads exceed it, 15 by more than 0.25 points, at most by 0.71. Against the split-half lead: mean absolute difference 0.1, mean signed difference -0.04 (mean lead: observed 0.1, split -0.02, estimate 0.02; largest split-half lead 0.8).

**Ceiling.** Exceeded by 0 of 156. A one-point lead is out of reach below an AUROC of 0.58-0.64 (by cell); the best score of a cell is above that floor in 11 of 12 cells, where the ceiling is 0.93-3.02 points. Observed leads reach 4% of the ceiling in the median and at most 48% (145 scores with a ceiling above 0.2 points).

Largest observed lead 0.82; largest help-vs-harm AUROC 0.748; AUROC that the estimate needs for +1 point 0.81-0.91.

| reader | benchmark | helps % | harms % | AUROC floor for +1 point | AUROC needed (estimate) | best AUROC of any score | observed / split / estimate / ceiling |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | evqa | 32.9 | 4.7 | 0.606 | 0.881 | 0.735 (factorised_post) | 0.18 / -0.17 / 0.05 / 2.22 |
| Qwen3-VL-4B-Instruct | infoseek | 14.2 | 4.6 | 0.609 | 0.823 | 0.728 (vgate_full) | 0.8 / 0.58 / 0.29 / 2.1 |
| Qwen3-VL-4B-Instruct | okvqa | 4.9 | 24.5 | 0.602 | 0.857 | 0.68 (error_post) | 0.25 / 0.22 / 0.02 / 1.76 |
| Qwen2.5-VL-3B-Instruct | evqa | 31.0 | 4.7 | 0.607 | 0.879 | 0.724 (factorised_post) | 0.04 / -0.08 / 0.04 / 2.08 |
| Qwen2.5-VL-3B-Instruct | infoseek | 14.0 | 4.4 | 0.613 | 0.829 | 0.696 (probe_benefit) | 0.19 / 0.06 / 0.13 / 1.74 |
| Qwen2.5-VL-3B-Instruct | okvqa | 5.6 | 19.3 | 0.589 | 0.813 | 0.679 (error_post) | 0.17 / 0.02 / 0.08 / 2.02 |
| InternVL3_5-4B-HF | evqa | 37.8 | 4.1 | 0.623 | 0.908 | 0.748 (benefit_post) | 0.0 / -0.07 / 0.03 / 2.02 |
| InternVL3_5-4B-HF | infoseek | 13.7 | 3.5 | 0.641 | 0.868 | 0.668 (probe_benefit) | 0.14 / -0.33 / 0.02 / 1.19 |
| InternVL3_5-4B-HF | okvqa | 6.1 | 23.8 | 0.582 | 0.818 | 0.748 (error_post) | 0.77 / 0.57 / 0.35 / 3.02 |
| Qwen3-VL-8B-Instruct | evqa | 33.6 | 5.3 | 0.595 | 0.867 | 0.74 (factorised_post) | 0.04 / -0.23 / 0.09 / 2.53 |
| Qwen3-VL-8B-Instruct | infoseek | 16.1 | 3.8 | 0.632 | 0.868 | 0.623 (probe_error) | 0.0 / -0.06 / 0.0 / 0.93 |
| Qwen3-VL-8B-Instruct | okvqa | 4.0 | 21.7 | 0.626 | 0.88 | 0.713 (signal_lp_min) | 0.0 / -0.07 / 0.04 / 1.7 |

CVQA (main reader, held-out): helps 8.3%, harms 14.0%; AUROC floor for +1 point 0.56, needed at the estimate 0.692. benefit_full: AUROC 0.626, observed best +0.73, estimate +0.30, ceiling +2.09; benefit_post: AUROC 0.62, observed best +0.67, estimate +0.26, ceiling +1.99; benefit_pre: AUROC 0.603, observed best +0.47, estimate +0.14, ceiling +1.72; after_reading: AUROC 0.747, observed best +1.68, estimate +1.75, ceiling +4.11.

Published gates (main reader, held-out), help-vs-harm AUROC of the gate's score / floor / needed at the estimate: chen/evqa 0.429 / 0.606 / 0.881; chen/infoseek 0.625 / 0.609 / 0.823; chen/okvqa 0.497 / 0.602 / 0.857; urag/evqa 0.502 / 0.606 / 0.881; urag/infoseek 0.575 / 0.609 / 0.823; urag/okvqa 0.48 / 0.602 / 0.857.

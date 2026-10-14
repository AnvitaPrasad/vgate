# PREREG v6e — which decision is worth a gate once the evidence is re-ranked (pilot; E-VQA development split, n = 1,000; 80% image-cluster intervals)

A: skip or read the retriever's five. B: skip or read the re-ranked five. C: the retriever's five or the re-ranked five. F0: VGate-full's features; F1: F0 and eight image-similarity features. Rule: take the action iff P(help) - P(harm) > 0, out of fold.

| reader | decision | helps % | harms % | oracle lead | AUROC floor / needed | fit | AUROC | lead [80%] | estimate | ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | A | 31.7 | 5.5 | 5.5 | 0.591 / 0.857 | F0 logistic | 0.685 | -0.70 [-1.37, -0.10] | +0.01 | +2.04 |
| Qwen3-VL-4B-Instruct | A | 31.7 | 5.5 | 5.5 | 0.591 / 0.857 | F0 boosted | 0.73 | -0.10 [-0.59, +0.39] | +0.08 | +2.52 |
| Qwen3-VL-4B-Instruct | A | 31.7 | 5.5 | 5.5 | 0.591 / 0.857 | F1 logistic | 0.696 | -0.40 [-1.15, +0.31] | +0.02 | +2.15 |
| Qwen3-VL-4B-Instruct | A | 31.7 | 5.5 | 5.5 | 0.591 / 0.857 | F1 boosted | 0.737 | -0.20 [-0.68, +0.30] | +0.11 | +2.61 |
| Qwen3-VL-4B-Instruct | B | 36.9 | 4.3 | 4.3 | 0.616 / 0.9 | F0 logistic | 0.68 | -0.90 [-1.43, -0.40] | +0.00 | +1.54 |
| Qwen3-VL-4B-Instruct | B | 36.9 | 4.3 | 4.3 | 0.616 / 0.9 | F0 boosted | 0.675 | -0.40 [-0.67, -0.19] | +0.00 | +1.51 |
| Qwen3-VL-4B-Instruct | B | 36.9 | 4.3 | 4.3 | 0.616 / 0.9 | F1 logistic | 0.681 | -1.20 [-1.77, -0.64] | +0.00 | +1.56 |
| Qwen3-VL-4B-Instruct | B | 36.9 | 4.3 | 4.3 | 0.616 / 0.9 | F1 boosted | 0.674 | -0.40 [-0.64, -0.18] | +0.00 | +1.50 |
| Qwen3-VL-4B-Instruct | C | 9.5 | 3.1 | 3.1 | 0.661 / 0.866 | F0 logistic | 0.59 | -0.90 [-1.44, -0.32] | +0.00 | +0.56 |
| Qwen3-VL-4B-Instruct | C | 9.5 | 3.1 | 3.1 | 0.661 / 0.866 | F0 boosted | 0.546 | -0.40 [-0.76, +0.00] | +0.00 | +0.29 |
| Qwen3-VL-4B-Instruct | C | 9.5 | 3.1 | 3.1 | 0.661 / 0.866 | F1 logistic | 0.603 | -0.60 [-1.12, +0.00] | +0.00 | +0.64 |
| Qwen3-VL-4B-Instruct | C | 9.5 | 3.1 | 3.1 | 0.661 / 0.866 | F1 boosted | 0.535 | -0.60 [-1.02, -0.20] | +0.00 | +0.22 |
| Qwen2.5-VL-3B-Instruct | A | 30.6 | 6.7 | 6.7 | 0.575 / 0.824 | F0 logistic | 0.724 | +0.10 [-0.59, +0.80] | +0.16 | +3.00 |
| Qwen2.5-VL-3B-Instruct | A | 30.6 | 6.7 | 6.7 | 0.575 / 0.824 | F0 boosted | 0.718 | +0.40 [-0.10, +0.93] | +0.13 | +2.92 |
| Qwen2.5-VL-3B-Instruct | A | 30.6 | 6.7 | 6.7 | 0.575 / 0.824 | F1 logistic | 0.718 | -0.60 [-1.36, +0.10] | +0.13 | +2.92 |
| Qwen2.5-VL-3B-Instruct | A | 30.6 | 6.7 | 6.7 | 0.575 / 0.824 | F1 boosted | 0.704 | +0.10 [-0.32, +0.52] | +0.09 | +2.73 |
| Qwen2.5-VL-3B-Instruct | B | 36.0 | 6.3 | 6.3 | 0.579 / 0.846 | F0 logistic | 0.72 | -0.20 [-0.86, +0.43] | +0.07 | +2.78 |
| Qwen2.5-VL-3B-Instruct | B | 36.0 | 6.3 | 6.3 | 0.579 / 0.846 | F0 boosted | 0.724 | -0.40 [-0.84, +0.00] | +0.08 | +2.82 |
| Qwen2.5-VL-3B-Instruct | B | 36.0 | 6.3 | 6.3 | 0.579 / 0.846 | F1 logistic | 0.731 | -0.60 [-1.29, +0.10] | +0.10 | +2.92 |
| Qwen2.5-VL-3B-Instruct | B | 36.0 | 6.3 | 6.3 | 0.579 / 0.846 | F1 boosted | 0.736 | -0.30 [-0.73, +0.10] | +0.12 | +2.97 |
| Qwen2.5-VL-3B-Instruct | C | 9.6 | 3.8 | 3.8 | 0.632 / 0.824 | F0 logistic | 0.558 | -1.20 [-1.82, -0.60] | +0.00 | +0.44 |
| Qwen2.5-VL-3B-Instruct | C | 9.6 | 3.8 | 3.8 | 0.632 / 0.824 | F0 boosted | 0.596 | -0.70 [-1.18, -0.20] | +0.00 | +0.73 |
| Qwen2.5-VL-3B-Instruct | C | 9.6 | 3.8 | 3.8 | 0.632 / 0.824 | F1 logistic | 0.583 | -0.90 [-1.53, -0.29] | +0.00 | +0.63 |
| Qwen2.5-VL-3B-Instruct | C | 9.6 | 3.8 | 3.8 | 0.632 / 0.824 | F1 boosted | 0.601 | -0.60 [-1.07, -0.10] | +0.00 | +0.77 |

Decision C, Qwen3-VL-4B-Instruct: questions the switch harms / helps by where the gold passage is: in both 16 / 41; lost 3 / 0; gained 2 / 40; in neither 10 / 14.
Decision C, Qwen2.5-VL-3B-Instruct: questions the switch harms / helps by where the gold passage is: in both 24 / 48; lost 4 / 0; gained 6 / 28; in neither 4 / 20.

GO (a B or C fit of the main reader leads by a point with the interval above zero, positive on the second reader): none.

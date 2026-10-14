# PREREG v6f — the planner's forecast for two further decisions, on the held-out split (preregistered; E-VQA, gates fitted on development data; 95% image-cluster intervals)

A: skip or read the retriever's five. B: skip or read the re-ranked five. C: the retriever's five or the re-ranked five. F0: VGate-full's features; F1: F0 and eight image-similarity features. Rule: take the action iff P(help) - P(harm) > 0, fitted on development data.

| reader | decision | helps % | harms % | oracle lead | AUROC floor / needed | fit | AUROC | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | A | 32.9 | 4.7 | 4.7 | 0.606 / 0.881 | F0 logistic | 0.73 | -0.40 [-0.86, +0.04] | +0.04 | +2.17 |
| Qwen3-VL-4B-Instruct | A | 32.9 | 4.7 | 4.7 | 0.606 / 0.881 | F0 boosted | 0.718 | -0.36 [-0.69, -0.08] | +0.03 | +2.06 |
| Qwen3-VL-4B-Instruct | A | 32.9 | 4.7 | 4.7 | 0.606 / 0.881 | F1 logistic | 0.76 | -0.04 [-0.52, +0.45] | +0.11 | +2.46 |
| Qwen3-VL-4B-Instruct | A | 32.9 | 4.7 | 4.7 | 0.606 / 0.881 | F1 boosted | 0.742 | -0.22 [-0.54, +0.11] | +0.07 | +2.29 |
| Qwen3-VL-4B-Instruct | B | 36.7 | 4.1 | 4.1 | 0.622 / 0.905 | F0 logistic | 0.728 | -0.40 [-0.73, -0.07] | +0.02 | +1.87 |
| Qwen3-VL-4B-Instruct | B | 36.7 | 4.1 | 4.1 | 0.622 / 0.905 | F0 boosted | 0.709 | -0.40 [-0.67, -0.19] | +0.01 | +1.72 |
| Qwen3-VL-4B-Instruct | B | 36.7 | 4.1 | 4.1 | 0.622 / 0.905 | F1 logistic | 0.738 | -0.22 [-0.58, +0.15] | +0.02 | +1.95 |
| Qwen3-VL-4B-Instruct | B | 36.7 | 4.1 | 4.1 | 0.622 / 0.905 | F1 boosted | 0.735 | -0.15 [-0.36, +0.04] | +0.02 | +1.93 |
| Qwen3-VL-4B-Instruct | C | 8.3 | 3.8 | 3.8 | 0.631 / 0.807 | F0 logistic | 0.525 | -0.55 [-1.09, +0.03] | +0.00 | +0.19 |
| Qwen3-VL-4B-Instruct | C | 8.3 | 3.8 | 3.8 | 0.631 / 0.807 | F0 boosted | 0.516 | -0.29 [-0.66, +0.04] | +0.00 | +0.12 |
| Qwen3-VL-4B-Instruct | C | 8.3 | 3.8 | 3.8 | 0.631 / 0.807 | F1 logistic | 0.566 | -0.87 [-1.51, -0.27] | +0.00 | +0.51 |
| Qwen3-VL-4B-Instruct | C | 8.3 | 3.8 | 3.8 | 0.631 / 0.807 | F1 boosted | 0.552 | -0.36 [-0.71, -0.03] | +0.00 | +0.40 |
| Qwen2.5-VL-3B-Instruct | A | 31.0 | 4.7 | 4.7 | 0.607 / 0.879 | F0 logistic | 0.72 | -0.33 [-0.87, +0.25] | +0.03 | +2.05 |
| Qwen2.5-VL-3B-Instruct | A | 31.0 | 4.7 | 4.7 | 0.607 / 0.879 | F0 boosted | 0.712 | -0.15 [-0.64, +0.34] | +0.02 | +1.97 |
| Qwen2.5-VL-3B-Instruct | A | 31.0 | 4.7 | 4.7 | 0.607 / 0.879 | F1 logistic | 0.733 | -0.47 [-1.00, +0.08] | +0.05 | +2.17 |
| Qwen2.5-VL-3B-Instruct | A | 31.0 | 4.7 | 4.7 | 0.607 / 0.879 | F1 boosted | 0.72 | -0.22 [-0.68, +0.21] | +0.03 | +2.04 |
| Qwen2.5-VL-3B-Instruct | B | 34.8 | 4.3 | 4.3 | 0.617 / 0.897 | F0 logistic | 0.721 | -0.44 [-1.06, +0.21] | +0.02 | +1.90 |
| Qwen2.5-VL-3B-Instruct | B | 34.8 | 4.3 | 4.3 | 0.617 / 0.897 | F0 boosted | 0.738 | +0.07 [-0.30, +0.48] | +0.03 | +2.04 |
| Qwen2.5-VL-3B-Instruct | B | 34.8 | 4.3 | 4.3 | 0.617 / 0.897 | F1 logistic | 0.721 | -0.44 [-1.01, +0.17] | +0.02 | +1.89 |
| Qwen2.5-VL-3B-Instruct | B | 34.8 | 4.3 | 4.3 | 0.617 / 0.897 | F1 boosted | 0.737 | +0.15 [-0.24, +0.57] | +0.03 | +2.03 |
| Qwen2.5-VL-3B-Instruct | C | 8.5 | 4.3 | 4.3 | 0.617 / 0.782 | F0 logistic | 0.491 | -0.62 [-1.11, -0.11] | +0.00 | +0.00 |
| Qwen2.5-VL-3B-Instruct | C | 8.5 | 4.3 | 4.3 | 0.617 / 0.782 | F0 boosted | 0.527 | -0.22 [-0.59, +0.18] | +0.00 | +0.24 |
| Qwen2.5-VL-3B-Instruct | C | 8.5 | 4.3 | 4.3 | 0.617 / 0.782 | F1 logistic | 0.575 | -0.40 [-0.90, +0.12] | +0.00 | +0.65 |
| Qwen2.5-VL-3B-Instruct | C | 8.5 | 4.3 | 4.3 | 0.617 / 0.782 | F1 boosted | 0.59 | -0.22 [-0.65, +0.24] | +0.01 | +0.77 |

Decision C, Qwen3-VL-4B-Instruct: questions the switch harms / helps by where the gold passage is: in both 49 / 99; lost 17 / 3; gained 5 / 86; in neither 34 / 39.
Decision C, Qwen2.5-VL-3B-Instruct: questions the switch harms / helps by where the gold passage is: in both 56 / 122; lost 16 / 2; gained 6 / 68; in neither 40 / 42.

Forecasts: F1 held; F2 held; F3 held. F1 without rounding the shares: NOT held (largest difference 2.009 points; allowed 2.0).

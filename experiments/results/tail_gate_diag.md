# Diagnosis of PREREG v10: per-query prediction or the identity of the question string? (exploratory; Qwen3-VL-4B-Instruct, held-out)

Lead over the benchmark's better fixed policy in points, 95% image-cluster intervals; rule: read iff P(help) - P(harm) > 0. A question string needs 3 training questions to get its own policy.

## evqa: n = 2750, helps 32.9%, harms 4.7%; AUROC floor 0.606, needed at the estimate 0.881; 2139 question strings, 6.7% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.718 | 93.7 | -0.18 [-0.59, +0.17] | 0.03 | 2.06 |
| logistic, string folds | 0.699 | 92.5 | -0.80 [-1.28, -0.39] | 0.01 | 1.89 |
| logistic, image folds, 1,000 labels | 0.691 | 90.5 | -0.65 [-1.19, -0.14] | 0.01 | 1.8 |
| boosted, image folds | 0.734 | 94.1 | -0.18 [-0.57, +0.19] | 0.05 | 2.21 |
| boosted, string folds | 0.742 | 94.7 | -0.51 [-0.88, -0.14] | 0.07 | 2.28 |
| boosted, image folds, 1,000 labels | 0.708 | 95.6 | -0.40 [-0.75, -0.07] | 0.02 | 1.97 |
| told the string (image folds) |  | 92.4 | -0.73 [-1.30, -0.18] |  |  |

Gate minus told-the-string (image folds): logistic +0.55 [-0.11, +1.19]; boosted +0.55 [-0.04, +1.15].

## infoseek: n = 3637, helps 14.2%, harms 4.6%; AUROC floor 0.609, needed at the estimate 0.823; 390 question strings, 69.2% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.779 | 83.6 | +1.21 [+0.74, +1.76] | 0.62 | 2.57 |
| logistic, string folds | 0.669 | 79.8 | -0.25 [-0.90, +0.40] | 0.06 | 1.55 |
| logistic, image folds, 1,000 labels | 0.722 | 84.8 | +0.41 [-0.14, +0.99] | 0.26 | 2.04 |
| boosted, image folds | 0.857 | 85.1 | +1.46 [+0.89, +2.06] | 1.37 | 3.28 |
| boosted, string folds | 0.68 | 88.0 | +0.08 [-0.30, +0.48] | 0.09 | 1.65 |
| boosted, image folds, 1,000 labels | 0.753 | 90.0 | +1.04 [+0.56, +1.51] | 0.43 | 2.32 |
| told the string (image folds) |  | 60.5 | +0.77 [+0.00, +1.53] |  |  |

Gate minus told-the-string (image folds): logistic +0.44 [-0.22, +1.14]; boosted +0.69 [+0.08, +1.35].

String folds over 10 random assignments: logistic lead -0.56 on average (-1.10 to -0.05), AUROC 0.603 (0.511-0.662); boosted lead -0.35 on average (-1.02 to +0.08), AUROC 0.656 (0.635-0.678).

## okvqa: n = 4046, helps 4.9%, harms 24.5%; AUROC floor 0.602, needed at the estimate 0.857; 3744 question strings, 0.3% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.65 | 6.7 | -0.17 [-0.52, +0.17] | 0.0 | 1.47 |
| logistic, string folds | 0.65 | 6.0 | +0.20 [-0.07, +0.47] | 0.0 | 1.47 |
| logistic, image folds, 1,000 labels | 0.611 | 9.9 | -0.94 [-1.41, -0.49] | 0.0 | 1.09 |
| boosted, image folds | 0.634 | 4.3 | -0.15 [-0.44, +0.12] | 0.0 | 1.31 |
| boosted, string folds | 0.621 | 4.1 | +0.00 [-0.32, +0.30] | 0.0 | 1.19 |
| boosted, image folds, 1,000 labels | 0.586 | 3.4 | -0.42 [-0.74, -0.10] | 0.0 | 0.84 |
| told the string (image folds) |  | 0.2 | -0.07 [-0.17, +0.00] |  |  |

Gate minus told-the-string (image folds): logistic -0.10 [-0.44, +0.22]; boosted -0.07 [-0.37, +0.22].


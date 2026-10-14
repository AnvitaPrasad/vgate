# Diagnosis of PREREG v10: per-query prediction or the identity of the question string? (exploratory; Qwen2.5-VL-3B-Instruct, held-out)

Lead over the benchmark's better fixed policy in points, 95% image-cluster intervals; rule: read iff P(help) - P(harm) > 0. A question string needs 3 training questions to get its own policy.

## evqa: n = 2750, helps 31.0%, harms 4.7%; AUROC floor 0.607, needed at the estimate 0.879; 2139 question strings, 6.7% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.725 | 89.2 | +0.00 [-0.43, +0.44] | 0.04 | 2.1 |
| logistic, string folds | 0.741 | 89.8 | +0.18 [-0.25, +0.63] | 0.07 | 2.24 |
| logistic, image folds, 1,000 labels | 0.683 | 89.3 | -1.27 [-1.88, -0.69] | 0.01 | 1.7 |
| boosted, image folds | 0.72 | 92.6 | +0.04 [-0.33, +0.42] | 0.03 | 2.05 |
| boosted, string folds | 0.736 | 93.3 | +0.00 [-0.37, +0.40] | 0.06 | 2.2 |
| boosted, image folds, 1,000 labels | 0.678 | 94.7 | -0.36 [-0.75, +0.00] | 0.0 | 1.66 |
| told the string (image folds) |  | 88.7 | -0.98 [-1.55, -0.41] |  |  |

Gate minus told-the-string (image folds): logistic +0.98 [+0.32, +1.64]; boosted +1.02 [+0.42, +1.61].

## infoseek: n = 3637, helps 14.0%, harms 4.4%; AUROC floor 0.613, needed at the estimate 0.829; 390 question strings, 69.2% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.679 | 88.3 | -0.19 [-0.78, +0.33] | 0.08 | 1.59 |
| logistic, string folds | 0.561 | 84.1 | -1.46 [-2.08, -0.86] | 0.0 | 0.54 |
| logistic, image folds, 1,000 labels | 0.637 | 83.4 | -0.77 [-1.40, -0.19] | 0.01 | 1.21 |
| boosted, image folds | 0.761 | 89.2 | +0.47 [-0.08, +1.01] | 0.45 | 2.31 |
| boosted, string folds | 0.514 | 89.5 | -0.77 [-1.25, -0.29] | 0.0 | 0.13 |
| boosted, image folds, 1,000 labels | 0.732 | 91.3 | +0.44 [-0.06, +0.93] | 0.28 | 2.06 |
| told the string (image folds) |  | 60.7 | +0.11 [-0.58, +0.81] |  |  |

Gate minus told-the-string (image folds): logistic -0.30 [-0.94, +0.30]; boosted +0.36 [-0.25, +0.97].

String folds over 10 random assignments: logistic lead -1.36 on average (-1.84 to -0.85), AUROC 0.548 (0.522-0.61); boosted lead -0.65 on average (-1.07 to -0.27), AUROC 0.533 (0.47-0.595).

## okvqa: n = 4046, helps 5.6%, harms 19.3%; AUROC floor 0.589, needed at the estimate 0.813; 3744 question strings, 0.3% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.627 | 8.8 | -0.37 [-0.79, +0.02] | 0.01 | 1.43 |
| logistic, string folds | 0.628 | 8.2 | -0.10 [-0.49, +0.30] | 0.01 | 1.45 |
| logistic, image folds, 1,000 labels | 0.588 | 13.2 | -1.19 [-1.76, -0.62] | 0.0 | 0.99 |
| boosted, image folds | 0.619 | 6.9 | -0.25 [-0.67, +0.15] | 0.0 | 1.34 |
| boosted, string folds | 0.626 | 6.2 | -0.02 [-0.42, +0.35] | 0.0 | 1.42 |
| boosted, image folds, 1,000 labels | 0.592 | 7.0 | -0.44 [-0.89, +0.00] | 0.0 | 1.04 |
| told the string (image folds) |  | 0.3 | -0.05 [-0.12, +0.00] |  |  |

Gate minus told-the-string (image folds): logistic -0.32 [-0.74, +0.10]; boosted -0.20 [-0.62, +0.20].


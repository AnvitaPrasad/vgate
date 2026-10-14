# Diagnosis of PREREG v10: per-query prediction or the identity of the question string? (exploratory; Qwen3-VL-8B-Instruct, held-out)

Lead over the benchmark's better fixed policy in points, 95% image-cluster intervals; rule: read iff P(help) - P(harm) > 0. A question string needs 3 training questions to get its own policy.

## evqa: n = 2750, helps 33.6%, harms 5.3%; AUROC floor 0.595, needed at the estimate 0.867; 2139 question strings, 6.7% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.721 | 89.6 | +0.00 [-0.50, +0.48] | 0.05 | 2.33 |
| logistic, string folds | 0.724 | 90.0 | +0.11 [-0.35, +0.57] | 0.05 | 2.36 |
| logistic, image folds, 1,000 labels | 0.687 | 86.8 | -0.98 [-1.66, -0.32] | 0.01 | 1.97 |
| boosted, image folds | 0.729 | 94.1 | +0.04 [-0.37, +0.46] | 0.06 | 2.42 |
| boosted, string folds | 0.724 | 93.8 | -0.18 [-0.61, +0.20] | 0.05 | 2.36 |
| boosted, image folds, 1,000 labels | 0.702 | 93.2 | -0.33 [-0.85, +0.16] | 0.02 | 2.13 |
| told the string (image folds) |  | 91.6 | -0.95 [-1.46, -0.45] |  |  |

Gate minus told-the-string (image folds): logistic +0.95 [+0.32, +1.60]; boosted +0.98 [+0.39, +1.58].

## infoseek: n = 3637, helps 16.1%, harms 3.8%; AUROC floor 0.632, needed at the estimate 0.868; 390 question strings, 69.2% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.715 | 87.8 | -0.08 [-0.46, +0.32] | 0.08 | 1.63 |
| logistic, string folds | 0.584 | 85.4 | -1.21 [-1.70, -0.74] | 0.0 | 0.64 |
| logistic, image folds, 1,000 labels | 0.662 | 84.8 | -1.02 [-1.53, -0.49] | 0.01 | 1.23 |
| boosted, image folds | 0.81 | 90.9 | +0.47 [+0.08, +0.89] | 0.51 | 2.35 |
| boosted, string folds | 0.649 | 90.5 | -0.77 [-1.20, -0.36] | 0.0 | 1.13 |
| boosted, image folds, 1,000 labels | 0.759 | 91.0 | -0.08 [-0.48, +0.30] | 0.23 | 1.97 |
| told the string (image folds) |  | 65.9 | -0.55 [-1.19, +0.08] |  |  |

Gate minus told-the-string (image folds): logistic +0.47 [-0.14, +1.10]; boosted +1.02 [+0.46, +1.62].

String folds over 10 random assignments: logistic lead -1.19 on average (-1.68 to -0.63), AUROC 0.61 (0.566-0.639); boosted lead -0.69 on average (-1.15 to -0.36), AUROC 0.659 (0.621-0.691).

## okvqa: n = 4046, helps 4.0%, harms 21.7%; AUROC floor 0.626, needed at the estimate 0.88; 3744 question strings, 0.3% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.595 | 4.5 | -0.35 [-0.64, -0.02] | 0.0 | 0.75 |
| logistic, string folds | 0.615 | 4.5 | -0.40 [-0.69, -0.10] | 0.0 | 0.92 |
| logistic, image folds, 1,000 labels | 0.579 | 8.0 | -0.67 [-1.09, -0.27] | 0.0 | 0.63 |
| boosted, image folds | 0.617 | 3.1 | -0.44 [-0.74, -0.15] | 0.0 | 0.93 |
| boosted, string folds | 0.605 | 2.6 | -0.27 [-0.52, -0.02] | 0.0 | 0.84 |
| boosted, image folds, 1,000 labels | 0.531 | 2.5 | -0.30 [-0.54, -0.05] | 0.0 | 0.24 |
| told the string (image folds) |  | 0.1 | -0.02 [-0.07, +0.00] |  |  |

Gate minus told-the-string (image folds): logistic -0.32 [-0.64, +0.00]; boosted -0.42 [-0.72, -0.12].


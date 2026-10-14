# Diagnosis of PREREG v10: per-query prediction or the identity of the question string? (exploratory; InternVL3_5-4B-HF, held-out)

Lead over the benchmark's better fixed policy in points, 95% image-cluster intervals; rule: read iff P(help) - P(harm) > 0. A question string needs 3 training questions to get its own policy.

## evqa: n = 2750, helps 37.8%, harms 4.1%; AUROC floor 0.623, needed at the estimate 0.908; 2139 question strings, 6.7% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.718 | 95.3 | -0.51 [-0.92, -0.11] | 0.01 | 1.77 |
| logistic, string folds | 0.745 | 95.2 | -0.18 [-0.51, +0.15] | 0.03 | 1.99 |
| logistic, image folds, 1,000 labels | 0.694 | 91.0 | -1.38 [-1.97, -0.80] | 0.0 | 1.58 |
| boosted, image folds | 0.744 | 97.1 | -0.15 [-0.51, +0.19] | 0.03 | 1.99 |
| boosted, string folds | 0.771 | 96.6 | +0.25 [-0.11, +0.61] | 0.07 | 2.21 |
| boosted, image folds, 1,000 labels | 0.735 | 96.1 | -0.25 [-0.64, +0.11] | 0.02 | 1.91 |
| told the string (image folds) |  | 93.4 | -1.24 [-1.81, -0.75] |  |  |

Gate minus told-the-string (image folds): logistic +0.73 [+0.11, +1.35]; boosted +1.09 [+0.55, +1.69].

## infoseek: n = 3637, helps 13.7%, harms 3.5%; AUROC floor 0.641, needed at the estimate 0.868; 390 question strings, 69.2% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.709 | 86.5 | +0.08 [-0.39, +0.58] | 0.09 | 1.48 |
| logistic, string folds | 0.62 | 84.5 | -0.71 [-1.25, -0.14] | 0.0 | 0.85 |
| logistic, image folds, 1,000 labels | 0.672 | 82.6 | -0.38 [-0.91, +0.16] | 0.02 | 1.22 |
| boosted, image folds | 0.769 | 91.0 | +0.77 [+0.37, +1.19] | 0.3 | 1.91 |
| boosted, string folds | 0.634 | 92.0 | -0.30 [-0.73, +0.13] | 0.0 | 0.95 |
| boosted, image folds, 1,000 labels | 0.709 | 90.7 | -0.08 [-0.49, +0.33] | 0.09 | 1.48 |
| told the string (image folds) |  | 59.6 | -0.93 [-1.60, -0.29] |  |  |

Gate minus told-the-string (image folds): logistic +1.02 [+0.35, +1.68]; boosted +1.70 [+1.13, +2.32].

String folds over 10 random assignments: logistic lead -0.49 on average (-0.82 to -0.14), AUROC 0.646 (0.615-0.691); boosted lead +0.07 on average (-0.36 to +0.41), AUROC 0.654 (0.608-0.687).

## okvqa: n = 4046, helps 6.1%, harms 23.8%; AUROC floor 0.582, needed at the estimate 0.818; 3744 question strings, 0.3% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.77 | 18.7 | +0.67 [+0.15, +1.21] | 0.51 | 3.28 |
| logistic, string folds | 0.761 | 18.9 | +0.87 [+0.32, +1.39] | 0.44 | 3.17 |
| logistic, image folds, 1,000 labels | 0.724 | 19.5 | +0.35 [-0.25, +0.91] | 0.21 | 2.72 |
| boosted, image folds | 0.755 | 16.6 | +0.67 [+0.20, +1.14] | 0.4 | 3.11 |
| boosted, string folds | 0.77 | 16.8 | +0.82 [+0.30, +1.31] | 0.52 | 3.29 |
| boosted, image folds, 1,000 labels | 0.665 | 12.2 | +0.15 [-0.27, +0.54] | 0.03 | 2.0 |
| told the string (image folds) |  | 0.0 | -0.02 [-0.07, +0.00] |  |  |

Gate minus told-the-string (image folds): logistic +0.69 [+0.15, +1.24]; boosted +0.69 [+0.22, +1.16].


# Diagnosis of PREREG v10: per-query prediction or the identity of the question string? (exploratory; Qwen3-VL-4B-Instruct, held-out)

Lead over the benchmark's better fixed policy in points, 95% image-cluster intervals; rule: read iff P(help) - P(harm) > 0. A question string needs 3 training questions to get its own policy.

## cvqa: n = 8373, helps 8.3%, harms 14.0%; AUROC floor 0.56, needed at the estimate 0.692; 6738 question strings, 3.4% of questions in strings used ten times or more

| policy | AUROC (help vs harm) | read % | lead [95%] | estimate | ceiling |
|---|---|---|---|---|---|
| logistic, image folds | 0.638 | 15.9 | +0.84 [+0.28, +1.36] | 0.41 | 2.29 |
| logistic, string folds | 0.629 | 16.2 | +0.56 [-0.04, +1.09] | 0.33 | 2.15 |
| logistic, image folds, 1,000 labels | 0.573 | 20.6 | +0.17 [-0.40, +0.71] | 0.02 | 1.21 |
| boosted, image folds | 0.621 | 15.2 | +0.48 [-0.06, +0.99] | 0.27 | 2.01 |
| boosted, string folds | 0.605 | 15.6 | +0.24 [-0.30, +0.76] | 0.16 | 1.75 |
| boosted, image folds, 1,000 labels | 0.553 | 15.8 | -0.18 [-0.64, +0.28] | 0.0 | 0.88 |
| told the string (image folds) |  | 1.0 | -0.08 [-0.18, +0.00] |  |  |

Gate minus told-the-string (image folds): logistic +0.92 [+0.35, +1.46]; boosted +0.56 [+0.04, +1.09].


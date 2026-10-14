# CVQA: choosing the answer after reading (PREREG v7c) — dev split, n = 2001, 80% image-cluster intervals

Accuracy: never-retrieve 65.72, always-retrieve 60.67, the rule 66.52, best of the two answers 75.96.
The rule uses the evidence answer on 51.6% of questions; it keeps 67.3% of the cases where retrieval helps and avoids 60.1% of the cases where it harms. Tokens relative to always-retrieve: 1.29.

- rule − never-retrieve (the better fixed policy): +0.80 [-0.37, +2.07]
- rule − always-retrieve: +5.85 [+4.76, +6.97]
- the probability comparison alone, without the names-an-option condition − never-retrieve: +0.80 [-0.37, +2.07] (the evidence reply names no option and is still the more confident one on 0.0% of questions)

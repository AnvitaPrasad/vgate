# CVQA: choosing the answer after reading (PREREG v7c) — confirm split, n = 8373, 95% image-cluster intervals

Accuracy: never-retrieve 65.58, always-retrieve 59.94, the rule 67.2, VGate-full 65.96, best of the two answers 73.9.
The rule uses the evidence answer on 33.1% of questions; it keeps 71.9% of the cases where retrieval helps and avoids 68.8% of the cases where it harms. Tokens relative to always-retrieve: 1.24.

- rule − never-retrieve (the better fixed policy): +1.62 [+0.89, +2.42]
- rule − always-retrieve: +7.26 [+6.50, +8.03]
- the probability comparison alone, without the names-an-option condition − never-retrieve: +0.76 [-0.01, +1.57] (the evidence reply names no option and is still the more confident one on 3.0% of questions)
- rule − VGate-full: +1.24 [+0.59, +1.90]
- rule − VGate-post (not a registered comparison): +1.03 [+0.36, +1.72]

## Expectations (PREREG v7c)

- S1: held
- S2: held

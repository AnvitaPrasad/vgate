# CVQA: choosing the answer after reading (PREREG v7c) — confirm split, n = 8373, 95% image-cluster intervals — FALLBACK LABELS (a reply that names no option counts as the closed-book answer for every policy)

Accuracy: never-retrieve 65.58, always-retrieve 63.07, the rule 67.2, VGate-full 66.44, best of the two answers 73.9.
The rule uses the evidence answer on 33.1% of questions; it keeps 71.9% of the cases where retrieval helps and avoids 59.8% of the cases where it harms. Tokens relative to always-retrieve: 1.24.

- rule − never-retrieve (the better fixed policy): +1.62 [+0.89, +2.42]
- rule − always-retrieve: +4.13 [+3.47, +4.77]
- the probability comparison alone, without the names-an-option condition − never-retrieve: +1.62 [+0.89, +2.42] (the evidence reply names no option and is still the more confident one on 3.0% of questions)
- rule − VGate-full: +0.76 [+0.14, +1.44]
- rule − VGate-post (not a registered comparison): +0.55 [-0.08, +1.19]

## Expectations (PREREG v7c)

- S1: held
- S2: held

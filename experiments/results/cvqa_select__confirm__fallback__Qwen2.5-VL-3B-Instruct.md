# CVQA: choosing the answer after reading (PREREG v7c) — confirm split, n = 8373, 95% image-cluster intervals — FALLBACK LABELS (a reply that names no option counts as the closed-book answer for every policy)

Accuracy: never-retrieve 64.45, always-retrieve 60.61, the rule 65.32, VGate-full 65.29, best of the two answers 74.63.
The rule uses the evidence answer on 50.8% of questions; it keeps 67.8% of the cases where retrieval helps and avoids 57.0% of the cases where it harms. Tokens relative to always-retrieve: 1.29.

- rule − never-retrieve (the better fixed policy): +0.87 [+0.05, +1.73]
- rule − always-retrieve: +4.71 [+3.95, +5.48]
- the probability comparison alone, without the names-an-option condition − never-retrieve: +0.87 [+0.05, +1.73] (the evidence reply names no option and is still the more confident one on 0.0% of questions)
- rule − VGate-full: +0.02 [-0.66, +0.72]
- rule − VGate-post (not a registered comparison): +0.11 [-0.54, +0.82]

## Expectations (PREREG v7c)

- S1: NOT held
- S2: NOT held

# CVQA: choosing the answer after reading (PREREG v7c) — dev split, n = 2001, 80% image-cluster intervals

Accuracy: never-retrieve 66.27, always-retrieve 60.72, the rule 68.77, best of the two answers 74.81.
The rule uses the evidence answer on 31.9% of questions; it keeps 74.3% of the cases where retrieval helps and avoids 72.7% of the cases where it harms. Tokens relative to always-retrieve: 1.24.

- rule − never-retrieve (the better fixed policy): +2.50 [+1.42, +3.66]
- rule − always-retrieve: +8.05 [+7.01, +9.11]
- the probability comparison alone, without the names-an-option condition − never-retrieve: +1.70 [+0.57, +2.89] (the evidence reply names no option and is still the more confident one on 2.8% of questions)

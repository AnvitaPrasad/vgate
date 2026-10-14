# CVQA, a fourth benchmark (PREREG v7) — dev split, n = 2001 (gates fitted on 2001 development questions, out of fold; 80% image-cluster intervals)

Retrieval helps on 8.55% of questions and hurts on 10.64%; an oracle gate is 8.55 points above the better fixed policy. Labels: an evidence reply that names no option falls back to the closed-book answer.

| Policy | Accuracy | Read (%) |
|---|---|---|
| never retrieve | 66.27 | 0.0 |
| always retrieve | 64.17 | 100.0 |
| better fixed policy (chosen on development data) | 66.27 | 0.0 |
| told the category (10) | 67.57 | 35.5 |
| category predicted from the gate's features | 67.52 | 36.9 |
| told the country-language subset (39) | 66.47 | 35.0 |
| subset predicted from the gate's features | 66.82 | 36.7 |
| error gate, free + retriever-score features | 67.47 | 22.6 |
| error gate, VGate-full's features | 67.32 | 30.9 |
| VGate-post | 67.02 | 22.5 |
| VGate-pre | 68.12 | 27.5 |
| VGate-full | 68.22 | 24.8 |
| oracle gate | 74.81 | 8.5 |

- VGate-full − better fixed policy: +1.95 [+0.90, +3.08]
- VGate-full − never: +1.95 [+0.90, +3.08]
- VGate-full − always: +4.05 [+2.96, +5.16]
- VGate-full − error gate with the same features: +0.90 [+0.20, +1.63]
- VGate-post − error gate with the same features: -0.45 [-1.10, +0.15]
- nAUC, benefit − error label (VGate-full's features): +0.78 [+0.36, +1.21]
- VGate-full − told the category: +0.65 [-0.15, +1.46]
- VGate-full − told the subset: +1.75 [+0.97, +2.59]
- VGate-full − category classifier: +0.70 [-0.10, +1.54]

## Expectations (PREREG v7)

- R1: held

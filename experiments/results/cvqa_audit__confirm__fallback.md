# CVQA, a fourth benchmark (PREREG v7) — confirm split, n = 8373 (gates fitted on 2001 development questions; 95% image-cluster intervals)

Retrieval helps on 8.32% of questions and hurts on 10.83%; an oracle gate is 8.32 points above the better fixed policy. Labels: an evidence reply that names no option falls back to the closed-book answer.

| Policy | Accuracy | Read (%) |
|---|---|---|
| never retrieve | 65.58 | 0.0 |
| always retrieve | 63.07 | 100.0 |
| better fixed policy (chosen on development data) | 65.58 | 0.0 |
| told the category (10) | 64.68 | 32.4 |
| category predicted from the gate's features | 65.11 | 34.0 |
| told the country-language subset (39) | 64.47 | 34.7 |
| subset predicted from the gate's features | 65.01 | 36.8 |
| error gate, free + retriever-score features | 65.89 | 22.3 |
| error gate, VGate-full's features | 66.43 | 31.2 |
| VGate-post | 66.65 | 23.0 |
| VGate-pre | 66.06 | 26.3 |
| VGate-full | 66.44 | 24.9 |
| oracle gate | 73.9 | 8.3 |

- VGate-full − better fixed policy: +0.86 [+0.23, +1.48]
- VGate-full − never: +0.86 [+0.23, +1.48]
- VGate-full − always: +3.37 [+2.55, +4.13]
- VGate-full − error gate with the same features: +0.01 [-0.57, +0.57]
- VGate-post − error gate with the same features: +0.76 [+0.28, +1.24]
- nAUC, benefit − error label (VGate-full's features): +0.78 [+0.39, +1.14]
- VGate-full − told the category: +1.76 [+1.10, +2.40]
- VGate-full − told the subset: +1.97 [+1.34, +2.58]
- VGate-full − category classifier: +1.33 [+0.69, +1.99]

## Expectations (PREREG v7)

- G1: NOT held
- G2: NOT held
- G3: held

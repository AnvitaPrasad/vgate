# CVQA, a fourth benchmark (PREREG v7) — confirm split, n = 8373 (gates fitted on 2001 development questions; 95% image-cluster intervals)

Retrieval helps on 8.32% of questions and hurts on 13.96%; an oracle gate is 8.32 points above the better fixed policy. The evidence reply names no option on 7.58% of questions (22.41% of the harm cases).

| Policy | Accuracy | Read (%) |
|---|---|---|
| never retrieve | 65.58 | 0.0 |
| always retrieve | 59.94 | 100.0 |
| better fixed policy (chosen on development data) | 65.58 | 0.0 |
| told the category (10) | 65.58 | 6.0 |
| category predicted from the gate's features | 65.68 | 7.3 |
| told the country-language subset (39) | 64.8 | 19.1 |
| subset predicted from the gate's features | 65.21 | 19.2 |
| error gate, free + retriever-score features | 65.01 | 22.3 |
| error gate, VGate-full's features | 64.97 | 31.2 |
| VGate-post | 66.18 | 13.6 |
| VGate-pre | 65.85 | 19.8 |
| VGate-full | 65.96 | 19.8 |
| oracle gate | 73.9 | 8.3 |

- VGate-full − better fixed policy: +0.38 [-0.20, +0.96]
- VGate-full − never: +0.38 [-0.20, +0.96]
- VGate-full − always: +6.02 [+5.13, +6.99]
- VGate-full − error gate with the same features: +0.99 [+0.34, +1.63]
- VGate-post − error gate with the same features: +1.17 [+0.63, +1.69]
- nAUC, benefit − error label (VGate-full's features): +1.70 [+1.27, +2.10]
- VGate-full − told the category: +0.38 [-0.21, +0.94]
- VGate-full − told the subset: +1.16 [+0.55, +1.75]
- VGate-full − category classifier: +0.29 [-0.31, +0.85]

## Expectations (PREREG v7)

- G1: NOT held
- G2: held
- G3: NOT held

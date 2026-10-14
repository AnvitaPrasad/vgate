# CVQA, a fourth benchmark (PREREG v7) — confirm split, n = 8373 (gates fitted on 2001 development questions; 95% image-cluster intervals)

Retrieval helps on 10.19% of questions and hurts on 14.02%; an oracle gate is 10.19 points above the better fixed policy. The evidence reply names no option on 0.0% of questions (0.0% of the harm cases).

| Policy | Accuracy | Read (%) |
|---|---|---|
| never retrieve | 64.45 | 0.0 |
| always retrieve | 60.61 | 100.0 |
| better fixed policy (chosen on development data) | 64.45 | 0.0 |
| told the category (10) | 64.45 | 0.0 |
| category predicted from the gate's features | 64.45 | 0.0 |
| told the country-language subset (39) | 64.19 | 11.3 |
| subset predicted from the gate's features | 64.53 | 8.0 |
| error gate, free + retriever-score features | 63.94 | 31.9 |
| error gate, VGate-full's features | 64.31 | 31.8 |
| VGate-post | 65.21 | 19.5 |
| VGate-pre | 64.58 | 25.5 |
| VGate-full | 65.29 | 30.3 |
| oracle gate | 74.63 | 10.2 |

- VGate-full − better fixed policy: +0.85 [+0.24, +1.52]
- VGate-full − never: +0.85 [+0.24, +1.52]
- VGate-full − always: +4.68 [+3.71, +5.65]
- VGate-full − error gate with the same features: +0.98 [+0.37, +1.60]
- VGate-post − error gate with the same features: +1.27 [+0.63, +1.91]
- nAUC, benefit − error label (VGate-full's features): +0.95 [+0.58, +1.32]
- VGate-full − told the category: +0.85 [+0.24, +1.52]
- VGate-full − told the subset: +1.10 [+0.48, +1.73]
- VGate-full − category classifier: +0.85 [+0.24, +1.52]

## Expectations (PREREG v7)

- G1: NOT held
- G2: held
- G3: held

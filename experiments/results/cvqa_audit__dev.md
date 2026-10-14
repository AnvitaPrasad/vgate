# CVQA, a fourth benchmark (PREREG v7) — dev split, n = 2001 (gates fitted on 2001 development questions, out of fold; 80% image-cluster intervals)

Retrieval helps on 8.55% of questions and hurts on 14.09%; an oracle gate is 8.55 points above the better fixed policy. The evidence reply names no option on 7.55% of questions (24.47% of the harm cases).

| Policy | Accuracy | Read (%) |
|---|---|---|
| never retrieve | 66.27 | 0.0 |
| always retrieve | 60.72 | 100.0 |
| better fixed policy (chosen on development data) | 66.27 | 0.0 |
| told the category (10) | 65.12 | 17.7 |
| category predicted from the gate's features | 65.62 | 15.6 |
| told the country-language subset (39) | 66.42 | 19.4 |
| subset predicted from the gate's features | 66.92 | 18.6 |
| error gate, free + retriever-score features | 66.32 | 22.6 |
| error gate, VGate-full's features | 65.37 | 30.9 |
| VGate-post | 66.37 | 13.7 |
| VGate-pre | 67.32 | 21.4 |
| VGate-full | 67.47 | 19.6 |
| oracle gate | 74.81 | 8.5 |

- VGate-full − better fixed policy: +1.20 [+0.20, +2.30]
- VGate-full − never: +1.20 [+0.20, +2.30]
- VGate-full − always: +6.75 [+5.49, +8.00]
- VGate-full − error gate with the same features: +2.10 [+1.31, +2.93]
- VGate-post − error gate with the same features: +0.05 [-0.65, +0.72]
- nAUC, benefit − error label (VGate-full's features): +1.93 [+1.43, +2.42]
- VGate-full − told the category: +2.35 [+1.38, +3.35]
- VGate-full − told the subset: +1.05 [+0.25, +1.91]
- VGate-full − category classifier: +1.85 [+0.89, +2.89]

## Expectations (PREREG v7)

- R1: held

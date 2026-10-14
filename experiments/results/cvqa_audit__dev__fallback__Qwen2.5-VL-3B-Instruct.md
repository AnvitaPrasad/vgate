# CVQA, a fourth benchmark (PREREG v7) — dev split, n = 2001 (gates fitted on 2001 development questions, out of fold; 80% image-cluster intervals)

Retrieval helps on 10.24% of questions and hurts on 15.29%; an oracle gate is 10.24 points above the better fixed policy. Labels: an evidence reply that names no option falls back to the closed-book answer.

| Policy | Accuracy | Read (%) |
|---|---|---|
| never retrieve | 65.72 | 0.0 |
| always retrieve | 60.67 | 100.0 |
| better fixed policy (chosen on development data) | 65.72 | 0.0 |
| told the category (10) | 65.47 | 1.6 |
| category predicted from the gate's features | 65.47 | 1.4 |
| told the country-language subset (39) | 63.02 | 17.5 |
| subset predicted from the gate's features | 64.57 | 16.1 |
| error gate, free + retriever-score features | 65.32 | 31.2 |
| error gate, VGate-full's features | 64.77 | 31.3 |
| VGate-post | 67.27 | 19.1 |
| VGate-pre | 64.42 | 28.1 |
| VGate-full | 65.32 | 30.5 |
| oracle gate | 75.96 | 10.2 |

- VGate-full − better fixed policy: -0.40 [-1.25, +0.44]
- VGate-full − never: -0.40 [-1.25, +0.44]
- VGate-full − always: +4.65 [+3.20, +6.06]
- VGate-full − error gate with the same features: +0.55 [-0.29, +1.41]
- VGate-post − error gate with the same features: +1.95 [+1.09, +2.79]
- nAUC, benefit − error label (VGate-full's features): +0.50 [-0.17, +1.10]
- VGate-full − told the category: -0.15 [-0.97, +0.70]
- VGate-full − told the subset: +2.30 [+1.30, +3.24]
- VGate-full − category classifier: -0.15 [-0.97, +0.67]

## Expectations (PREREG v7)

- R1: held

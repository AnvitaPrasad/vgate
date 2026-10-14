# Passage selection with the released critic (PREREG v5 pilot, dev split, 80% intervals)

| | evqa | infoseek | okvqa |
|---|---|---|---|
| never | 26.7 | 17.3 | 59.4 |
| always (top-5) | 52.9 | 25.7 | 38.6 |
| critic-pick (no fitted parameter) | 55.1 | 25.5 | 49.7 |
| … answers closed-book (%) | 8.7 | 36.2 | 35.8 |
| passages accepted per question | 2.83 | 1.35 | 1.81 |
| tokens per query | 3003.9 | 2703.8 | 2710.0 |
| tokens, always | 1074.6 | 1019.3 | 984.7 |
| gold in top-5 (%) | 57.7 | 68.1 | — |
| arg-max critic is the gold passage (%) | 71.2 | 55.4 | — |
| AUROC for the gold passage | 0.815 | 0.749 | — |
| recall at the card's threshold | 91.0 | 60.8 | — |
| specificity at the card's threshold | 47.8 | 78.2 | — |
| critic-pick − always | +2.2 [+0.5, +3.9] | -0.2 [-1.7, +1.2] | +11.1 [+9.4, +12.8] |
| critic-pick − never | +28.4 [+26.2, +30.6] | +8.2 [+6.6, +9.8] | -9.7 [-11.3, -8.1] |

Macro over the benchmarks: better fixed policy per benchmark 46.0; critic-pick 43.43 (-2.6 [-3.5, -1.6]); one benefit-trained selector without dataset identity 45.2 (-0.8 [-1.6, +0.1]; {'evqa': 53.8, 'infoseek': 26.4, 'okvqa': 55.4}).

## Expectations (PREREG v5)

- P1: NOT held
- P2: NOT held
- P3: NOT held

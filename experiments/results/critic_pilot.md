# Passage selection with the released critic (PREREG v5 pilot, dev split, 80% intervals)

| | evqa | infoseek | okvqa |
|---|---|---|---|
| never | 26.7 | 17.3 | 59.4 |
| always (top-5) | 52.9 | 25.7 | 38.6 |
| critic-pick (no fitted parameter) | 55.8 | 23.9 | 49.1 |
| … answers closed-book (%) | 4.9 | 62.8 | 61.5 |
| passages accepted per question | 3.14 | 0.48 | 0.66 |
| tokens per query | 4915.7 | 4531.4 | 4553.5 |
| tokens, always | 1074.6 | 1019.3 | 984.7 |
| gold in top-5 (%) | 57.7 | 68.1 | — |
| arg-max critic is the gold passage (%) | 71.8 | 52.6 | — |
| AUROC for the gold passage | 0.878 | 0.78 | — |
| recall at the card's threshold | 95.5 | 30.5 | — |
| specificity at the card's threshold | 41.4 | 93.7 | — |
| critic-pick − always | +2.9 [+1.1, +4.7] | -1.8 [-3.5, -0.1] | +10.5 [+8.7, +12.3] |
| critic-pick − never | +29.1 [+26.9, +31.3] | +6.6 [+5.2, +7.9] | -10.3 [-11.7, -9.0] |

Macro over the benchmarks: better fixed policy per benchmark 46.0; critic-pick 42.93 (-3.1 [-4.0, -2.1]); one benefit-trained selector without dataset identity 45.63 (-0.4 [-1.2, +0.5]; {'evqa': 54.7, 'infoseek': 26.0, 'okvqa': 56.2}).

## Expectations (PREREG v5)

- P1: NOT held
- P2: NOT held
- P3: NOT held

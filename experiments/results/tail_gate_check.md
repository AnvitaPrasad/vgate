# PREREG v10 — gradient-boosted trees against logistic regression on the main reader's held-out log (exploratory)

75 signals; five folds out of fold, image groups kept together; rule: read iff P(help) - P(harm) > 0; 95% image-cluster intervals.

| benchmark | helps % | harms % | AUROC floor / needed | model | AUROC | read % | rule's lead [95%] | best threshold | estimate | ceiling |
|---|---|---|---|---|---|---|---|---|---|---|
| evqa | 32.9 | 4.7 | 0.606 / 0.881 | logistic | 0.718 | 93.7 | -0.18 [-0.59, +0.17] | +0.00 | +0.03 | +2.06 |
| evqa | 32.9 | 4.7 | 0.606 / 0.881 | boosted | 0.734 | 94.1 | -0.18 [-0.57, +0.19] | +0.11 | +0.05 | +2.21 |
| infoseek | 14.2 | 4.6 | 0.609 / 0.823 | logistic | 0.779 | 83.6 | +1.21 [+0.74, +1.76] | +1.35 | +0.62 | +2.57 |
| infoseek | 14.2 | 4.6 | 0.609 / 0.823 | boosted | 0.857 | 85.1 | +1.46 [+0.89, +2.06] | +1.51 | +1.37 | +3.28 |
| okvqa | 4.9 | 24.5 | 0.602 / 0.857 | logistic | 0.65 | 6.7 | -0.17 [-0.52, +0.17] | +0.00 | +0.00 | +1.47 |
| okvqa | 4.9 | 24.5 | 0.602 / 0.857 | boosted | 0.634 | 4.3 | -0.15 [-0.44, +0.12] | +0.02 | +0.00 | +1.31 |

T1 (boosted rule's lead below one point on every benchmark): NOT held. T2 (AUROC within 0.03 of the logistic model's on every benchmark): NOT held. Replication on the other readers triggered: yes.

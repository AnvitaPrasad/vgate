# PREREG v13 — a larger reader: Qwen3-VL-32B-Instruct, 4-bit, on 1498 held-out questions (498 evqa, 500 infoseek, 500 okvqa)

| benchmark | never | always | helps % | harms % | AUROC floor | needed | null lead | gate (out of fold): AUROC | its lead over the better fixed policy [95%] |
|---|---|---|---|---|---|---|---|---|---|
| evqa | 24.5 | 49.8 | 31.7 | 6.4 | 0.578 | 0.833 | 0.05 | 0.625 | -3.61 [-5.48, -1.88] |
| infoseek | 12.0 | 22.8 | 14.4 | 3.6 | 0.639 | 0.869 | 0.06 | 0.625 | -1.00 [-2.22, +0.00] |
| okvqa | 62.4 | 32.6 | 3.0 | 32.8 | 0.667 | 0.937 | 0.02 | 0.747 | -1.60 [-3.00, -0.20] |

Mixed traffic (accuracy = mean over the three benchmarks; gates fitted out of fold on this log):

| policy | accuracy | read % | vs told the benchmark [95%] |
|---|---|---|---|
| never retrieve | 33.0 | 0.0 | -12.0 [-14.1, -10.0] |
| always retrieve | 35.1 | 100.0 | -9.9 [-11.4, -8.5] |
| fixed per segment (needs the label) | 45.0 | 67.0 | +0.0 [+0.0, +0.0] |
| segment predicted | 44.3 | 67.0 | -0.7 [-1.2, -0.3] |
| error gate | 41.5 | 72.0 | -3.5 [-4.6, -2.5] |
| benefit gate | 43.5 | 56.0 | -1.5 [-2.4, -0.5] |
| benefit gate per segment (needs the label) | 42.9 | 57.0 | -2.1 [-3.0, -1.2] |
| unseen segment: error gate | 37.5 | 71.0 | -7.5 [-9.0, -5.9] |
| unseen segment: benefit gate | 33.5 | 64.0 | -11.5 [-13.4, -9.5] |
| oracle gate | 49.3 | 16.0 | +4.3 [+3.4, +5.4] |

Expectations: Q1 held; Q2 held; Q3 held; Q4 held.

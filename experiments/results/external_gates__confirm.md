# Published gates through the audit (PREREG v8) — confirm split, reader Qwen3-VL-4B-Instruct, 95% image-cluster intervals

Each gate is used as released (prompt or checkpoint, published threshold); its decisions are applied to the reader's closed-book and top-5 answers.

## Search-need prompt of Chen et al. (reader's reply: true / false)

| | evqa | infoseek | okvqa |
|---|---|---|---|
| reads (%) | 23.0 | 48.3 | 40.7 |
| accuracy following the gate | 28.07 | 24.14 | 52.2 |
| never | 23.05 | 17.76 | 59.59 |
| always | 51.2 | 27.36 | 39.94 |
| better fixed policy | 51.2 | 27.36 | 59.59 |
| oracle gate | 55.93 | 31.95 | 64.48 |
| decisions equal to its majority on the benchmark (%) | 77.0 | 51.7 | 59.3 |
| AUROC of its score, help vs harm | 0.429 | 0.625 | 0.497 |
| AUROC of its score, closed-book answer wrong | 0.444 | 0.455 | 0.47 |
| probability mass on the expected words | 1.0 | 1.0 | 1.0 |
| reply not readable (%) | 0.0 | 0.0 | 0.0 |
| reply agrees with the probability rule (%) | 99.6 | 99.6 | 99.7 |
| gate − better fixed policy | -23.13 [-25.74, -20.30] | -3.22 [-4.21, -2.19] | -7.39 [-8.41, -6.40] |
| gate − random decisions at its read rate (exploratory) | -1.46 [-2.50, -0.33] | +1.74 [+1.00, +2.45] | +0.60 [-0.22, +1.41] |

Mixed traffic (mean of the three): gate 34.8, never 33.47, always 39.5, told the benchmark 46.05, oracle 50.79. Gate − told the benchmark -11.24 [-12.17, -10.20]; gate − always -4.70 [-5.65, -3.62]; gate − its own majority decision per benchmark +1.34 [+0.74, +1.99] (62.7% of decisions equal that majority); gate − random decisions at its read rate on each benchmark +0.29 [-0.20, +0.84] (exploratory, added after the results).

Expectations: X1 held, X2 held, X3 NOT held.

## UniversalRAG router prompt (reader's reply: no / a modality; text query only)

| | evqa | infoseek | okvqa |
|---|---|---|---|
| reads (%) | 99.5 | 98.8 | 96.2 |
| accuracy following the gate | 50.95 | 27.39 | 40.43 |
| never | 23.05 | 17.76 | 59.59 |
| always | 51.2 | 27.36 | 39.94 |
| better fixed policy | 51.2 | 27.36 | 59.59 |
| oracle gate | 55.93 | 31.95 | 64.48 |
| decisions equal to its majority on the benchmark (%) | 99.5 | 98.8 | 96.2 |
| AUROC of its score, help vs harm | 0.502 | 0.575 | 0.48 |
| AUROC of its score, closed-book answer wrong | 0.503 | 0.489 | 0.501 |
| probability mass on the expected words | 1.0 | 1.0 | 1.0 |
| reply not readable (%) | 0.0 | 0.0 | 0.0 |
| reply agrees with the probability rule (%) | 99.9 | 100.0 | 99.9 |
| gate − better fixed policy | -0.25 [-0.48, -0.04] | +0.03 [+0.00, +0.08] | -19.15 [-20.72, -17.60] |
| gate − random decisions at its read rate (exploratory) | -0.10 [-0.32, +0.11] | +0.15 [+0.11, +0.21] | -0.24 [-0.52, +0.03] |

Mixed traffic (mean of the three): gate 39.59, never 33.47, always 39.5, told the benchmark 46.05, oracle 50.79. Gate − told the benchmark -6.46 [-6.97, -5.94]; gate − always +0.09 [-0.03, +0.20]; gate − its own majority decision per benchmark +0.09 [-0.03, +0.20] (98.2% of decisions equal that majority); gate − random decisions at its read rate on each benchmark -0.07 [-0.19, +0.05] (exploratory, added after the results).

Expectations: X1 held, X2 held, X3 held.

## ReAG-Critic checkpoint, kept passage in the top-5 (after retrieval)

NOT_RUN on this split (share of questions scored: {'evqa': 0.0, 'infoseek': 0.0, 'okvqa': 0.0}).

## ReflectiVA checkpoint, [Retrieval] token

| | evqa | infoseek | okvqa |
|---|---|---|---|
| reads (%) | 100.0 | 100.0 | 78.7 |
| accuracy following the gate | 51.2 | 27.36 | 43.38 |
| never | 23.05 | 17.76 | 59.59 |
| always | 51.2 | 27.36 | 39.94 |
| better fixed policy | 51.2 | 27.36 | 59.59 |
| oracle gate | 55.93 | 31.95 | 64.48 |
| decisions equal to its majority on the benchmark (%) | 100.0 | 100.0 | 78.7 |
| AUROC of its score, help vs harm | 0.509 | 0.47 | 0.525 |
| AUROC of its score, closed-book answer wrong | 0.54 | 0.602 | 0.543 |
| probability mass on the expected words | 0.978 | 0.978 | 0.968 |
| reply not readable (%) | 100.0 | 100.0 | 100.0 |
| reply agrees with the probability rule (%) | nan | nan | nan |
| gate − better fixed policy | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -16.21 [-17.61, -14.83] |
| gate − random decisions at its read rate (exploratory) | +0.01 [+0.01, +0.01] | +0.00 [+0.00, +0.00] | -0.75 [-1.36, -0.10] |

Mixed traffic (mean of the three): gate 40.64, never 33.47, always 39.5, told the benchmark 46.05, oracle 50.79. Gate − told the benchmark -5.40 [-5.88, -4.92]; gate − always +1.15 [+0.94, +1.37]; gate − its own majority decision per benchmark +1.15 [+0.94, +1.37] (92.9% of decisions equal that majority); gate − random decisions at its read rate on each benchmark -0.25 [-0.44, -0.03] (exploratory, added after the results).

Expectations: X1 held, X2 held, X3 NOT held.


# Published gates through the audit (PREREG v8) — dev split, reader Qwen3-VL-4B-Instruct, 95% image-cluster intervals

Each gate is used as released (prompt or checkpoint, published threshold); its decisions are applied to the reader's closed-book and top-5 answers.

## Search-need prompt of Chen et al. (reader's reply: true / false)

| | evqa | infoseek | okvqa |
|---|---|---|---|
| reads (%) | 25.6 | 49.4 | 41.3 |
| accuracy following the gate | 31.0 | 23.0 | 51.1 |
| never | 26.7 | 17.3 | 59.4 |
| always | 52.9 | 25.7 | 38.6 |
| better fixed policy | 52.9 | 25.7 | 59.4 |
| oracle gate | 58.4 | 30.9 | 64.0 |
| decisions equal to its majority on the benchmark (%) | 74.4 | 50.6 | 58.7 |
| AUROC of its score, help vs harm | 0.404 | 0.62 | 0.504 |
| AUROC of its score, closed-book answer wrong | 0.41 | 0.416 | 0.473 |
| probability mass on the expected words | 1.0 | 1.0 | 1.0 |
| reply not readable (%) | 0.0 | 0.0 | 0.0 |
| reply agrees with the probability rule (%) | 99.9 | 99.6 | 99.7 |
| gate − better fixed policy | -21.90 [-25.15, -18.78] | -2.70 [-4.78, -0.68] | -8.30 [-10.39, -6.19] |
| gate − random decisions at its read rate (exploratory) | -2.41 [-3.97, -0.94] | +1.55 [+0.19, +2.99] | +0.29 [-1.34, +1.86] |

Mixed traffic (mean of the three): gate 35.03, never 34.47, always 39.07, told the benchmark 46.0, oracle 51.1. Gate − told the benchmark -10.97 [-12.40, -9.59]; gate − always -4.03 [-5.57, -2.52]; gate − its own majority decision per benchmark +0.57 [-0.52, +1.71] (61.2% of decisions equal that majority); gate − random decisions at its read rate on each benchmark -0.19 [-1.06, +0.72] (exploratory, added after the results).

Expectations: X1 held, X2 held, X3 held.

## UniversalRAG router prompt (reader's reply: no / a modality; text query only)

| | evqa | infoseek | okvqa |
|---|---|---|---|
| reads (%) | 99.7 | 98.7 | 95.5 |
| accuracy following the gate | 52.6 | 25.7 | 39.0 |
| never | 26.7 | 17.3 | 59.4 |
| always | 52.9 | 25.7 | 38.6 |
| better fixed policy | 52.9 | 25.7 | 59.4 |
| oracle gate | 58.4 | 30.9 | 64.0 |
| decisions equal to its majority on the benchmark (%) | 99.7 | 98.7 | 95.5 |
| AUROC of its score, help vs harm | 0.483 | 0.609 | 0.476 |
| AUROC of its score, closed-book answer wrong | 0.502 | 0.491 | 0.504 |
| probability mass on the expected words | 1.0 | 1.0 | 1.0 |
| reply not readable (%) | 0.0 | 0.0 | 0.0 |
| reply agrees with the probability rule (%) | 100.0 | 100.0 | 99.9 |
| gate − better fixed policy | -0.30 [-0.69, +0.00] | +0.00 [+0.00, +0.00] | -20.40 [-23.45, -17.32] |
| gate − random decisions at its read rate (exploratory) | -0.22 [-0.61, +0.08] | +0.11 [+0.07, +0.15] | -0.54 [-1.16, +0.16] |

Mixed traffic (mean of the three): gate 39.1, never 34.47, always 39.07, told the benchmark 46.0, oracle 51.1. Gate − told the benchmark -6.90 [-7.93, -5.83]; gate − always +0.03 [-0.21, +0.30]; gate − its own majority decision per benchmark +0.03 [-0.21, +0.30] (98.0% of decisions equal that majority); gate − random decisions at its read rate on each benchmark -0.22 [-0.46, +0.03] (exploratory, added after the results).

Expectations: X1 held, X2 held, X3 held.

## ReAG-Critic checkpoint, kept passage in the top-5 (after retrieval)

| | evqa | infoseek | okvqa |
|---|---|---|---|
| reads (%) | 95.1 | 37.2 | 38.5 |
| accuracy following the gate | 53.1 | 22.8 | 49.9 |
| never | 26.7 | 17.3 | 59.4 |
| always | 52.9 | 25.7 | 38.6 |
| better fixed policy | 52.9 | 25.7 | 59.4 |
| oracle gate | 58.4 | 30.9 | 64.0 |
| decisions equal to its majority on the benchmark (%) | 95.1 | 62.8 | 61.5 |
| AUROC of its score, help vs harm | 0.721 | 0.682 | 0.435 |
| AUROC of its score, closed-book answer wrong | 0.511 | 0.489 | 0.452 |
| gate − better fixed policy | +0.20 [-0.41, +0.86] | -2.90 [-5.16, -0.68] | -9.50 [-11.62, -7.42] |
| gate − random decisions at its read rate (exploratory) | +1.48 [+0.89, +2.10] | +2.38 [+0.99, +3.73] | -1.49 [-3.16, +0.11] |

Mixed traffic (mean of the three): gate 41.93, never 34.47, always 39.07, told the benchmark 46.0, oracle 51.1. Gate − told the benchmark -4.07 [-5.10, -3.03]; gate − always +2.87 [+1.72, +4.06]; gate − its own majority decision per benchmark -1.27 [-2.23, -0.32] (73.1% of decisions equal that majority); gate − random decisions at its read rate on each benchmark +0.79 [+0.04, +1.53] (exploratory, added after the results).

Expectations: X1 held, X2 held, X3 held.

## ReflectiVA checkpoint, [Retrieval] token

| | evqa | infoseek | okvqa |
|---|---|---|---|
| reads (%) | 100.0 | 100.0 | 79.9 |
| accuracy following the gate | 52.9 | 25.7 | 42.4 |
| never | 26.7 | 17.3 | 59.4 |
| always | 52.9 | 25.7 | 38.6 |
| better fixed policy | 52.9 | 25.7 | 59.4 |
| oracle gate | 58.4 | 30.9 | 64.0 |
| decisions equal to its majority on the benchmark (%) | 100.0 | 100.0 | 79.9 |
| AUROC of its score, help vs harm | 0.492 | 0.417 | 0.471 |
| AUROC of its score, closed-book answer wrong | 0.558 | 0.56 | 0.53 |
| probability mass on the expected words | 0.978 | 0.978 | 0.968 |
| reply not readable (%) | 100.0 | 100.0 | 100.0 |
| reply agrees with the probability rule (%) | nan | nan | nan |
| gate − better fixed policy | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -17.00 [-19.86, -14.09] |
| gate − random decisions at its read rate (exploratory) | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -0.38 [-1.64, +0.94] |

Mixed traffic (mean of the three): gate 40.33, never 34.47, always 39.07, told the benchmark 46.0, oracle 51.1. Gate − told the benchmark -5.67 [-6.61, -4.70]; gate − always +1.27 [+0.80, +1.77]; gate − its own majority decision per benchmark +1.27 [+0.80, +1.77] (93.3% of decisions equal that majority); gate − random decisions at its read rate on each benchmark -0.13 [-0.57, +0.33] (exploratory, added after the results).

Expectations: X1 held, X2 held, X3 NOT held.


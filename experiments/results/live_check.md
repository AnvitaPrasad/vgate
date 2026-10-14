# Live-pipeline system check (PREREG v2f)
Run: `experiments/runs/J9`, model Qwen3-VL-4B-Instruct; every pass executed for every query, one query at a time.

| dataset | n | closed-book answers = logged | u_pre corr. | u_pre sign agreement | search | read |
|---|---|---|---|---|---|---|
| okvqa | 400 | 99.8% | 1.0000 | 100.0% | 8% | 7% |
| evqa | 200 | 99.0% | 1.0000 | 100.0% | 90% | 80% |
| infoseek | 200 | 100.0% | 1.0000 | 100.0% | 87% | 78% |

| dataset | never | always (live retriever) | VGate (budget-free) | VGate − always | VGate − never | logged never / always (benchmark retriever) |
|---|---|---|---|---|---|---|
| okvqa | 58.0 [53.2, 63.0] | 38.0 [33.0, 43.0] | 57.8 [53.0, 62.7] | +19.8 [+15.0, +25.0] | -0.2 [-1.5, +1.0] | 58.0 / 38.8 |
| evqa | 22.5 [16.8, 28.4] | 15.0 [10.4, 20.0] | 19.0 [14.0, 24.5] | +4.0 [+1.5, +7.0] | -3.5 [-8.4, +1.1] | 23.0 / 51.5 |
| infoseek | 19.0 [13.9, 24.5] | 11.5 [7.5, 16.0] | 13.0 [8.6, 17.9] | +1.5 [-0.5, +3.9] | -6.0 [-10.3, -2.0] | 19.0 / 26.0 |

| dataset | closed-book s (median / p90) | caption s | retrieval s | evidence s | tokens: closed-book / caption / evidence / VGate |
|---|---|---|---|---|---|
| okvqa | 0.41 / 0.47 | 1.74 / 2.17 | 0.11 / 0.12 | 1.15 / 1.39 | 277 / 290 / 990 / 369 |
| evqa | 0.37 / 0.53 | 1.64 / 2.14 | 0.11 / 0.12 | 1.23 / 1.65 | 257 / 265 / 1025 / 1317 |
| infoseek | 0.42 / 0.53 | 1.88 / 2.34 | 0.11 / 0.12 | 1.33 / 1.63 | 264 / 277 / 1035 / 1301 |

OK-VQA retrieval with a VLM caption vs the dataset caption: top-5 overlap 38%; answer-string pseudo recall@5 61% (live) vs 60% (logged).

## Gates refit on the live system's own logs (PREREG v2g; out of fold, exploratory)

Retrieval with the live retriever helps / harms: okvqa 5.8% / 25.8%, evqa 4.5% / 12.0%, infoseek 2.0% / 9.5%.

| policy | okvqa acc (search / read) | evqa acc (search / read) | infoseek acc (search / read) | macro |
|---|---|---|---|---|
| never | 58.0 (0% / 0%) | 22.5 (0% / 0%) | 19.0 (0% / 0%) | 33.2 |
| always | 38.0 (100% / 100%) | 15.0 (100% / 100%) | 11.5 (100% / 100%) | 21.5 |
| transferred | 57.8 (8% / 7%) | 19.0 (90% / 80%) | 13.0 (87% / 78%) | 29.9 |
| refit | 57.5 (14% / 13%) | 24.0 (18% / 15%) | 18.0 (14% / 10%) | 33.2 |
| refit_pre_only | 57.0 (14% / 14%) | 23.5 (18% / 18%) | 16.5 (14% / 14%) | 32.3 |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| refit - transferred | +3.25 | +1.25 | +5.34 |
| refit - never | +0.00 | -1.00 | +1.01 |
| refit - always | +11.67 | +8.92 | +14.49 |
| transferred - never | -3.25 | -5.44 | -1.12 |
| refit_pre_only - never | -0.83 | -2.02 | +0.34 |

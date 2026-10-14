# Gates on a benchmark they were not fitted on (dev split; PREREG v4a)

Accuracy % (share of queries read %) on the benchmark named in the column; gates fitted on the dev sets of the other two.

## Qwen3-VL-4B-Instruct

| rule | evqa | infoseek | okvqa | macro |
|---|---|---|---|---|
| never | 26.7 (0) | 17.3 (0) | 59.4 (0) | 34.5 |
| always | 52.9 (100) | 25.7 (100) | 38.6 (100) | 39.1 |
| train_majority | 26.7 (0) | 25.7 (100) | 38.6 (100) | 30.3 |
| nearest_regime | 48.1 (74) | 24.7 (90) | 38.6 (100) | 37.1 |
| error_post | 50.6 (70) | 24.8 (63) | 40.2 (88) | 38.5 |
| error_full | 51.5 (88) | 25.0 (73) | 42.2 (75) | 39.6 |
| benefit_post | 39.4 (30) | 26.1 (80) | 39.2 (98) | 34.9 |
| benefit_pre | 38.0 (45) | 24.9 (84) | 39.1 (92) | 34.0 |
| benefit_full | 40.6 (44) | 25.8 (91) | 40.4 (89) | 35.6 |
| best_fixed | 52.9 | 25.7 | 59.4 | 46.0 |
| oracle | 58.4 (32) | 30.9 (14) | 64.0 (5) | 51.1 |

Macro differences (95%): error_full - benefit_full +4.0 [+2.9, +5.0]; error_post - benefit_post +3.6 [+2.8, +4.7]; nearest_regime - best_fixed -8.9 [-10.1, -7.8]; error_post - best_fixed -7.5 [-8.8, -6.3]; error_full - best_fixed -6.4 [-7.5, -5.3]; benefit_post - best_fixed -11.1 [-12.8, -9.5]; benefit_pre - best_fixed -12.0 [-13.4, -10.7]; benefit_full - best_fixed -10.4 [-11.9, -9.0].

## Qwen2.5-VL-3B-Instruct

| rule | evqa | infoseek | okvqa | macro |
|---|---|---|---|---|
| never | 24.6 (0) | 15.7 (0) | 61.5 (0) | 33.9 |
| always | 48.5 (100) | 25.6 (100) | 46.4 (100) | 40.2 |
| train_majority | 24.6 (0) | 25.6 (100) | 46.4 (100) | 32.2 |
| nearest_regime | 43.3 (73) | 25.0 (92) | 46.4 (100) | 38.2 |
| error_post | 47.6 (80) | 21.7 (68) | 51.1 (71) | 40.1 |
| error_full | 48.2 (90) | 22.5 (74) | 51.4 (64) | 40.7 |
| benefit_post | 36.1 (32) | 24.8 (82) | 46.8 (97) | 35.9 |
| benefit_pre | 39.6 (66) | 24.7 (87) | 47.9 (94) | 37.4 |
| benefit_full | 38.9 (57) | 24.9 (91) | 51.0 (80) | 38.3 |
| best_fixed | 48.5 | 25.6 | 61.5 | 45.2 |
| oracle | 55.2 (31) | 30.0 (14) | 68.0 (6) | 51.1 |

Macro differences (95%): error_full - benefit_full +2.4 [+1.5, +3.4]; error_post - benefit_post +4.2 [+3.2, +5.4]; nearest_regime - best_fixed -7.0 [-8.2, -5.7]; error_post - best_fixed -5.1 [-6.1, -4.0]; error_full - best_fixed -4.5 [-5.5, -3.4]; benefit_post - best_fixed -9.3 [-10.9, -7.8]; benefit_pre - best_fixed -7.8 [-9.1, -6.4]; benefit_full - best_fixed -6.9 [-8.1, -5.7].

## Expectations (PREREG v4a)

- L1: held
- L2: held
- L3: held

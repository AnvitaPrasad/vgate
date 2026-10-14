# Gates on a benchmark they were not fitted on (confirm split; PREREG v4a)

Accuracy % (share of queries read %) on the benchmark named in the column; gates fitted on the dev sets of the other two.

## Qwen3-VL-4B-Instruct

| rule | evqa | infoseek | okvqa | macro |
|---|---|---|---|---|
| never | 23.1 (0) | 17.8 (0) | 59.6 (0) | 33.5 |
| always | 51.2 (100) | 27.4 (100) | 39.9 (100) | 39.5 |
| train_majority | 23.1 (0) | 27.4 (100) | 39.9 (100) | 30.1 |
| nearest_regime | 47.5 (78) | 26.6 (90) | 39.9 (100) | 38.0 |
| error_post | 48.4 (77) | 25.1 (63) | 42.2 (89) | 38.6 |
| error_full | 50.4 (90) | 25.7 (73) | 43.7 (77) | 39.9 |
| benefit_post | 37.8 (35) | 26.8 (81) | 40.2 (98) | 34.9 |
| benefit_pre | 37.5 (45) | 26.2 (85) | 40.6 (92) | 34.8 |
| benefit_full | 39.3 (44) | 26.9 (90) | 41.9 (89) | 36.0 |
| best_fixed | 51.2 | 27.4 | 59.6 | 46.0 |
| oracle | 55.9 (33) | 31.9 (14) | 64.5 (5) | 50.8 |

Macro differences (95%): error_full - benefit_full +3.9 [+3.3, +4.5]; error_post - benefit_post +3.7 [+3.0, +4.3]; nearest_regime - best_fixed -8.0 [-8.6, -7.4]; error_post - best_fixed -7.5 [-8.1, -6.8]; error_full - best_fixed -6.1 [-6.7, -5.5]; benefit_post - best_fixed -11.1 [-11.9, -10.3]; benefit_pre - best_fixed -11.3 [-12.0, -10.6]; benefit_full - best_fixed -10.0 [-10.7, -9.3].

## Qwen2.5-VL-3B-Instruct

| rule | evqa | infoseek | okvqa | macro |
|---|---|---|---|---|
| never | 22.3 (0) | 17.0 (0) | 59.5 (0) | 32.9 |
| always | 48.6 (100) | 26.6 (100) | 45.9 (100) | 40.4 |
| train_majority | 22.3 (0) | 26.6 (100) | 45.9 (100) | 31.6 |
| nearest_regime | 44.5 (77) | 25.9 (90) | 45.9 (100) | 38.8 |
| error_post | 47.5 (84) | 24.0 (67) | 49.2 (72) | 40.2 |
| error_full | 48.2 (90) | 24.3 (74) | 50.5 (66) | 41.0 |
| benefit_post | 37.1 (37) | 25.6 (82) | 46.1 (97) | 36.3 |
| benefit_pre | 41.4 (67) | 25.4 (86) | 47.2 (93) | 38.0 |
| benefit_full | 41.0 (59) | 26.0 (89) | 49.2 (79) | 38.8 |
| best_fixed | 48.6 | 26.6 | 59.5 | 44.9 |
| oracle | 53.3 (31) | 31.0 (14) | 65.2 (6) | 49.8 |

Macro differences (95%): error_full - benefit_full +2.2 [+1.6, +2.8]; error_post - benefit_post +3.9 [+3.3, +4.5]; nearest_regime - best_fixed -6.1 [-6.8, -5.5]; error_post - best_fixed -4.7 [-5.2, -4.1]; error_full - best_fixed -3.9 [-4.4, -3.4]; benefit_post - best_fixed -8.6 [-9.3, -7.8]; benefit_pre - best_fixed -6.9 [-7.6, -6.2]; benefit_full - best_fixed -6.1 [-6.8, -5.5].

## Expectations (PREREG v4a)

- L1: held
- L2: held
- L3: held

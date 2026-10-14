# Gates on a benchmark they were not fitted on (confirm split; PREREG v4a)

Accuracy % (share of queries read %) on the benchmark named in the column; gates fitted on the dev sets of the other two.

## InternVL3_5-4B-HF

| rule | evqa | infoseek | okvqa | macro |
|---|---|---|---|---|
| never | 19.9 (0) | 9.5 (0) | 44.0 (0) | 24.4 |
| always | 53.6 (100) | 19.6 (100) | 26.3 (100) | 33.2 |
| train_majority | 19.9 (0) | 19.6 (100) | 26.3 (100) | 21.9 |
| nearest_regime | 39.0 (56) | 18.0 (76) | 26.3 (100) | 27.8 |
| error_post | 51.0 (84) | 18.7 (86) | 28.3 (90) | 32.7 |
| error_full | 52.4 (91) | 18.5 (90) | 28.5 (86) | 33.1 |
| benefit_post | 40.4 (49) | 18.7 (90) | 26.3 (100) | 28.4 |
| benefit_pre | 37.3 (45) | 17.9 (83) | 26.7 (98) | 27.3 |
| benefit_full | 40.6 (52) | 18.9 (91) | 27.7 (95) | 29.1 |
| best_fixed | 53.6 | 19.6 | 44.0 | 39.1 |
| oracle | 57.7 (38) | 23.1 (14) | 50.1 (6) | 43.6 |

Macro differences (95%): error_full - benefit_full +4.1 [+3.5, +4.6]; error_post - benefit_post +4.2 [+3.7, +4.8]; nearest_regime - best_fixed -11.3 [-12.0, -10.4]; error_post - best_fixed -6.4 [-7.0, -5.8]; error_full - best_fixed -6.0 [-6.5, -5.4]; benefit_post - best_fixed -10.6 [-11.4, -9.8]; benefit_pre - best_fixed -11.8 [-12.6, -10.9]; benefit_full - best_fixed -10.0 [-10.8, -9.2].

## Qwen3-VL-8B-Instruct

| rule | evqa | infoseek | okvqa | macro |
|---|---|---|---|---|
| never | 26.6 (0) | 19.3 (0) | 61.9 (0) | 35.9 |
| always | 54.8 (100) | 31.6 (100) | 44.3 (100) | 43.6 |
| train_majority | 26.6 (0) | 31.6 (100) | 44.3 (100) | 34.1 |
| nearest_regime | 51.5 (78) | 28.8 (84) | 44.3 (100) | 41.5 |
| error_post | 52.9 (76) | 28.1 (66) | 48.0 (76) | 43.0 |
| error_full | 53.8 (84) | 29.5 (80) | 47.6 (76) | 43.6 |
| benefit_post | 45.0 (43) | 30.4 (84) | 44.3 (100) | 39.9 |
| benefit_pre | 44.0 (49) | 30.9 (90) | 44.6 (95) | 39.8 |
| benefit_full | 44.4 (46) | 31.1 (92) | 44.6 (95) | 40.0 |
| best_fixed | 54.8 | 31.6 | 61.9 | 49.5 |
| oracle | 60.1 (34) | 35.4 (16) | 65.9 (4) | 53.8 |

Macro differences (95%): error_full - benefit_full +3.6 [+3.1, +4.1]; error_post - benefit_post +3.1 [+2.6, +3.7]; nearest_regime - best_fixed -8.0 [-8.6, -7.3]; error_post - best_fixed -6.5 [-7.1, -5.9]; error_full - best_fixed -5.8 [-6.4, -5.3]; benefit_post - best_fixed -9.6 [-10.3, -8.9]; benefit_pre - best_fixed -9.7 [-10.4, -8.9]; benefit_full - best_fixed -9.4 [-10.1, -8.7].

## Expectations (PREREG v4a)

- L1: held
- L2: held
- L3: held

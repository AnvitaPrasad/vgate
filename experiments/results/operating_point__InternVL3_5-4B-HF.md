# Operating points (PREREG v2e, exploratory): InternVL3_5-4B-HF, labels=official
Held-out accuracy (share of queries that read evidence). Rules are budget-free; nothing is tuned on held-out data.

## Per-dataset gates (fit on each dataset's dev split)

| rule | evqa acc (reads) | infoseek acc (reads) | okvqa acc (reads) | macro acc (reads) |
|---|---|---|---|---|
| never | 19.9 (0%) | 9.5 (0%) | 44.0 (0%) | 24.4 (0%) |
| always | 53.6 (100%) | 19.6 (100%) | 26.3 (100%) | 33.2 (100%) |
| fixed_by_dataset | 53.6 (100%) | 19.6 (100%) | 44.0 (0%) | 39.1 (67%) |
| router | 53.6 (100%) | 19.6 (100%) | 44.0 (0%) | 39.1 (67%) |
| error_rule | 52.5 (91%) | 19.5 (99%) | 41.2 (49%) | 37.7 (80%) |
| error_rule_post_only | 52.5 (91%) | 19.5 (99%) | 41.2 (49%) | 37.7 (80%) |
| error_rule_full | 52.8 (90%) | 19.4 (95%) | 40.7 (51%) | 37.6 (79%) |
| benefit_rule_post | 53.6 (100%) | 19.6 (99%) | 44.5 (16%) | 39.2 (72%) |
| benefit_rule_full | 53.0 (93%) | 19.5 (85%) | 44.5 (17%) | 39.0 (65%) |
| benefit_rule_pre | 53.0 (96%) | 19.6 (85%) | 43.5 (4%) | 38.7 (62%) |
| cascade | 52.9 (93%) | 19.3 (81%) | 43.7 (2%) | 38.6 (59%) |
| error_tuned | 53.6 (98%) | 19.5 (100%) | 44.7 (17%) | 39.3 (72%) |
| error_full_tuned | 53.7 (97%) | 19.7 (99%) | 44.4 (14%) | 39.2 (70%) |
| benefit_full_tuned | 53.6 (100%) | 19.7 (96%) | 44.0 (0%) | 39.1 (65%) |
| oracle | 57.7 (38%) | 23.1 (14%) | 50.1 (6%) | 43.6 (19%) |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| benefit_rule_post - error_rule | +1.53 | +1.16 | +1.90 |
| benefit_rule_post - always | +6.08 | +5.54 | +6.59 |
| benefit_rule_post - never | +14.80 | +13.71 | +15.80 |
| benefit_rule_full - error_rule | +1.26 | +0.83 | +1.67 |
| benefit_rule_full - always | +5.81 | +5.23 | +6.35 |
| benefit_rule_full - never | +14.53 | +13.49 | +15.49 |
| benefit_rule_pre - error_rule | +0.97 | +0.55 | +1.38 |
| benefit_rule_pre - always | +5.52 | +4.93 | +6.06 |
| benefit_rule_pre - never | +14.24 | +13.19 | +15.20 |
| benefit_full_tuned - error_tuned | -0.17 | -0.38 | +0.03 |
| benefit_rule_full - error_tuned | -0.30 | -0.59 | -0.02 |
| benefit_rule_full - error_rule_full | +1.34 | +0.92 | +1.75 |
| benefit_rule_full - router | -0.09 | -0.37 | +0.19 |
| benefit_rule_full - fixed_by_dataset | -0.09 | -0.37 | +0.19 |
| benefit_rule_pre - router | -0.38 | -0.61 | -0.16 |
| benefit_rule_pre - fixed_by_dataset | -0.38 | -0.61 | -0.16 |
| cascade - router | -0.45 | -0.73 | -0.18 |
| cascade - fixed_by_dataset | -0.45 | -0.73 | -0.18 |
| cascade - always | +5.45 | +4.85 | +6.02 |
| cascade - error_rule_full | +0.98 | +0.53 | +1.40 |
| benefit_full_tuned - error_full_tuned | -0.13 | -0.33 | +0.04 |

| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |
|---|---|---|---|
| evqa | +0.15 | -0.46 | +0.72 |
| infoseek | +0.08 | -0.50 | +0.64 |
| okvqa | +3.78 | +2.87 | +4.79 |

Search rates (macro): benefit_rule_pre 62%, cascade 62%, router 67%

## Mixed traffic (one gate fit on the pooled dev splits, no dataset identity)

| rule | evqa acc (reads) | infoseek acc (reads) | okvqa acc (reads) | macro acc (reads) |
|---|---|---|---|---|
| never | 19.9 (0%) | 9.5 (0%) | 44.0 (0%) | 24.4 (0%) |
| always | 53.6 (100%) | 19.6 (100%) | 26.3 (100%) | 33.2 (100%) |
| fixed_by_dataset | 53.6 (100%) | 19.6 (100%) | 44.0 (0%) | 39.1 (67%) |
| router | 52.1 (96%) | 19.4 (97%) | 43.4 (6%) | 38.3 (66%) |
| error_rule | 52.0 (88%) | 19.0 (93%) | 34.6 (69%) | 35.2 (84%) |
| error_rule_post_only | 52.0 (88%) | 19.0 (93%) | 34.6 (69%) | 35.2 (84%) |
| error_rule_full | 52.5 (90%) | 19.1 (95%) | 36.8 (58%) | 36.1 (81%) |
| benefit_rule_post | 50.1 (78%) | 18.7 (89%) | 39.4 (50%) | 36.1 (72%) |
| benefit_rule_full | 51.6 (88%) | 19.4 (88%) | 43.1 (16%) | 38.0 (64%) |
| benefit_rule_pre | 52.0 (91%) | 18.7 (89%) | 42.7 (14%) | 37.8 (65%) |
| cascade | 51.2 (87%) | 18.7 (85%) | 43.2 (11%) | 37.7 (61%) |
| error_tuned | 50.9 (83%) | 18.7 (88%) | 37.7 (56%) | 35.8 (76%) |
| error_full_tuned | 50.9 (84%) | 18.5 (92%) | 40.4 (42%) | 36.6 (73%) |
| benefit_full_tuned | 51.4 (87%) | 19.4 (87%) | 43.2 (14%) | 38.0 (63%) |
| oracle | 57.7 (38%) | 23.1 (14%) | 50.1 (6%) | 43.6 (19%) |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| benefit_rule_post - error_rule | +0.88 | +0.46 | +1.30 |
| benefit_rule_post - always | +2.91 | +2.36 | +3.47 |
| benefit_rule_post - never | +11.64 | +10.62 | +12.62 |
| benefit_rule_full - error_rule | +2.81 | +2.28 | +3.33 |
| benefit_rule_full - always | +4.84 | +4.24 | +5.42 |
| benefit_rule_full - never | +13.56 | +12.53 | +14.55 |
| benefit_rule_pre - error_rule | +2.59 | +2.01 | +3.14 |
| benefit_rule_pre - always | +4.62 | +4.03 | +5.20 |
| benefit_rule_pre - never | +13.35 | +12.34 | +14.34 |
| benefit_full_tuned - error_tuned | +2.22 | +1.71 | +2.71 |
| benefit_rule_full - error_tuned | +2.22 | +1.73 | +2.70 |
| benefit_rule_full - error_rule_full | +1.86 | +1.40 | +2.30 |
| benefit_rule_full - router | -0.27 | -0.60 | +0.06 |
| benefit_rule_full - fixed_by_dataset | -1.06 | -1.40 | -0.70 |
| benefit_rule_pre - router | -0.49 | -0.85 | -0.17 |
| benefit_rule_pre - fixed_by_dataset | -1.27 | -1.63 | -0.93 |
| cascade - router | -0.59 | -0.94 | -0.26 |
| cascade - fixed_by_dataset | -1.37 | -1.74 | -1.01 |
| cascade - always | +4.53 | +3.89 | +5.15 |
| cascade - error_rule_full | +1.55 | +1.03 | +2.02 |
| benefit_full_tuned - error_full_tuned | +1.38 | +0.94 | +1.82 |

| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |
|---|---|---|---|
| evqa | -0.91 | -1.61 | -0.23 |
| infoseek | +0.25 | -0.22 | +0.74 |
| okvqa | +6.25 | +5.14 | +7.32 |

Search rates (macro): benefit_rule_pre 65%, cascade 65%, router 66%
Router: dataset predicted from the closed-book hidden state with 89.7% accuracy.

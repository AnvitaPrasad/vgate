# Operating points (PREREG v2e, exploratory): Qwen3-VL-4B-Instruct, labels=official
Held-out accuracy (share of queries that read evidence). Rules are budget-free; nothing is tuned on held-out data.

## Per-dataset gates (fit on each dataset's dev split)

| rule | evqa acc (reads) | infoseek acc (reads) | okvqa acc (reads) | macro acc (reads) |
|---|---|---|---|---|
| never | 23.1 (0%) | 17.8 (0%) | 59.6 (0%) | 33.5 (0%) |
| always | 51.2 (100%) | 27.4 (100%) | 39.9 (100%) | 39.5 (100%) |
| fixed_by_dataset | 51.2 (100%) | 27.4 (100%) | 59.6 (0%) | 46.0 (67%) |
| router | 51.2 (100%) | 27.4 (100%) | 59.6 (0%) | 46.0 (67%) |
| error_rule | 49.2 (87%) | 27.3 (99%) | 56.2 (28%) | 44.2 (71%) |
| error_rule_post_only | 49.2 (87%) | 27.3 (99%) | 56.2 (28%) | 44.2 (71%) |
| error_rule_full | 49.9 (85%) | 27.1 (87%) | 55.4 (32%) | 44.2 (68%) |
| benefit_rule_post | 51.2 (98%) | 26.8 (94%) | 59.7 (2%) | 45.9 (65%) |
| benefit_rule_full | 50.8 (90%) | 27.8 (77%) | 59.2 (9%) | 45.9 (59%) |
| benefit_rule_pre | 50.8 (95%) | 28.0 (81%) | 58.9 (5%) | 45.9 (60%) |
| cascade | 50.7 (89%) | 27.6 (74%) | 59.3 (4%) | 45.9 (56%) |
| error_tuned | 51.1 (100%) | 27.4 (100%) | 59.6 (0%) | 46.0 (67%) |
| error_full_tuned | 51.2 (100%) | 27.2 (96%) | 59.6 (0%) | 46.0 (65%) |
| benefit_full_tuned | 51.2 (100%) | 27.8 (82%) | 59.6 (0%) | 46.2 (61%) |
| oracle | 55.9 (33%) | 31.9 (14%) | 64.5 (5%) | 50.8 (17%) |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| benefit_rule_post - error_rule | +1.67 | +1.28 | +2.06 |
| benefit_rule_post - always | +6.40 | +5.86 | +6.94 |
| benefit_rule_post - never | +12.43 | +11.34 | +13.41 |
| benefit_rule_full - error_rule | +1.71 | +1.26 | +2.14 |
| benefit_rule_full - always | +6.44 | +5.85 | +6.99 |
| benefit_rule_full - never | +12.47 | +11.41 | +13.44 |
| benefit_rule_pre - error_rule | +1.69 | +1.28 | +2.12 |
| benefit_rule_pre - always | +6.43 | +5.84 | +6.98 |
| benefit_rule_pre - never | +12.46 | +11.39 | +13.42 |
| benefit_full_tuned - error_tuned | +0.18 | -0.02 | +0.39 |
| benefit_rule_full - error_tuned | -0.07 | -0.38 | +0.21 |
| benefit_rule_full - error_rule_full | +1.78 | +1.35 | +2.22 |
| benefit_rule_full - router | -0.11 | -0.41 | +0.17 |
| benefit_rule_full - fixed_by_dataset | -0.11 | -0.41 | +0.17 |
| benefit_rule_pre - router | -0.12 | -0.38 | +0.11 |
| benefit_rule_pre - fixed_by_dataset | -0.12 | -0.38 | +0.11 |
| cascade - router | -0.20 | -0.50 | +0.08 |
| cascade - fixed_by_dataset | -0.20 | -0.50 | +0.08 |
| cascade - always | +6.35 | +5.75 | +6.94 |
| cascade - error_rule_full | +1.70 | +1.26 | +2.13 |
| benefit_full_tuned - error_full_tuned | +0.18 | -0.01 | +0.38 |

| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |
|---|---|---|---|
| evqa | +0.91 | +0.19 | +1.65 |
| infoseek | +0.63 | +0.03 | +1.26 |
| okvqa | +3.81 | +2.94 | +4.65 |

Search rates (macro): benefit_rule_pre 60%, cascade 60%, router 67%

## Mixed traffic (one gate fit on the pooled dev splits, no dataset identity)

| rule | evqa acc (reads) | infoseek acc (reads) | okvqa acc (reads) | macro acc (reads) |
|---|---|---|---|---|
| never | 23.1 (0%) | 17.8 (0%) | 59.6 (0%) | 33.5 (0%) |
| always | 51.2 (100%) | 27.4 (100%) | 39.9 (100%) | 39.5 (100%) |
| fixed_by_dataset | 51.2 (100%) | 27.4 (100%) | 59.6 (0%) | 46.0 (67%) |
| router | 50.6 (98%) | 27.3 (99%) | 58.9 (4%) | 45.6 (67%) |
| error_rule | 49.3 (80%) | 26.4 (83%) | 51.0 (48%) | 42.2 (70%) |
| error_rule_post_only | 49.3 (80%) | 26.4 (83%) | 51.0 (48%) | 42.2 (70%) |
| error_rule_full | 49.9 (86%) | 27.1 (89%) | 53.5 (35%) | 43.5 (70%) |
| benefit_rule_post | 46.3 (66%) | 26.6 (78%) | 55.8 (29%) | 42.9 (58%) |
| benefit_rule_full | 49.5 (84%) | 27.6 (82%) | 58.5 (12%) | 45.2 (60%) |
| benefit_rule_pre | 49.8 (90%) | 27.5 (84%) | 58.1 (10%) | 45.1 (62%) |
| cascade | 49.0 (82%) | 27.2 (78%) | 58.7 (7%) | 45.0 (56%) |
| error_tuned | 45.6 (66%) | 25.6 (64%) | 56.2 (25%) | 42.5 (52%) |
| error_full_tuned | 48.7 (80%) | 26.9 (84%) | 55.4 (27%) | 43.7 (64%) |
| benefit_full_tuned | 48.6 (78%) | 27.4 (74%) | 58.7 (8%) | 44.9 (53%) |
| oracle | 55.9 (33%) | 31.9 (14%) | 64.5 (5%) | 50.8 (17%) |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| benefit_rule_post - error_rule | +0.69 | +0.18 | +1.17 |
| benefit_rule_post - always | +3.41 | +2.76 | +4.02 |
| benefit_rule_post - never | +9.45 | +8.36 | +10.40 |
| benefit_rule_full - error_rule | +2.95 | +2.43 | +3.46 |
| benefit_rule_full - always | +5.67 | +5.09 | +6.25 |
| benefit_rule_full - never | +11.71 | +10.60 | +12.68 |
| benefit_rule_pre - error_rule | +2.90 | +2.36 | +3.43 |
| benefit_rule_pre - always | +5.62 | +5.05 | +6.18 |
| benefit_rule_pre - never | +11.65 | +10.59 | +12.64 |
| benefit_full_tuned - error_tuned | +2.44 | +1.91 | +2.95 |
| benefit_rule_full - error_tuned | +2.71 | +2.19 | +3.24 |
| benefit_rule_full - error_rule_full | +1.69 | +1.21 | +2.12 |
| benefit_rule_full - router | -0.44 | -0.81 | -0.08 |
| benefit_rule_full - fixed_by_dataset | -0.88 | -1.26 | -0.53 |
| benefit_rule_pre - router | -0.49 | -0.82 | -0.16 |
| benefit_rule_pre - fixed_by_dataset | -0.93 | -1.25 | -0.60 |
| cascade - router | -0.64 | -1.02 | -0.28 |
| cascade - fixed_by_dataset | -1.08 | -1.46 | -0.72 |
| cascade - always | +5.47 | +4.85 | +6.04 |
| cascade - error_rule_full | +1.48 | +1.00 | +1.95 |
| benefit_full_tuned - error_full_tuned | +1.23 | +0.74 | +1.67 |

| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |
|---|---|---|---|
| evqa | -0.40 | -1.26 | +0.42 |
| infoseek | +0.47 | -0.06 | +1.00 |
| okvqa | +4.99 | +4.13 | +5.86 |

Search rates (macro): benefit_rule_pre 62%, cascade 62%, router 67%
Router: dataset predicted from the closed-book hidden state with 91.1% accuracy.

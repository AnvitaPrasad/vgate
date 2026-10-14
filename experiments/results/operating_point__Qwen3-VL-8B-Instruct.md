# Operating points (PREREG v2e, exploratory): Qwen3-VL-8B-Instruct, labels=official
Held-out accuracy (share of queries that read evidence). Rules are budget-free; nothing is tuned on held-out data.

## Per-dataset gates (fit on each dataset's dev split)

| rule | evqa acc (reads) | infoseek acc (reads) | okvqa acc (reads) | macro acc (reads) |
|---|---|---|---|---|
| never | 26.5 (0%) | 19.3 (0%) | 61.9 (0%) | 35.9 (0%) |
| always | 54.8 (100%) | 31.6 (100%) | 44.3 (100%) | 43.6 (100%) |
| fixed_by_dataset | 54.8 (100%) | 31.6 (100%) | 61.9 (0%) | 49.5 (67%) |
| router | 54.8 (100%) | 31.6 (100%) | 61.9 (0%) | 49.5 (67%) |
| error_rule | 52.8 (83%) | 31.3 (98%) | 58.7 (26%) | 47.6 (69%) |
| error_rule_post_only | 52.8 (83%) | 31.3 (98%) | 58.7 (26%) | 47.6 (69%) |
| error_rule_full | 53.0 (83%) | 30.9 (88%) | 57.8 (30%) | 47.2 (67%) |
| benefit_rule_post | 54.9 (100%) | 31.6 (100%) | 61.7 (5%) | 49.4 (68%) |
| benefit_rule_full | 55.1 (92%) | 30.9 (85%) | 61.2 (8%) | 49.1 (61%) |
| benefit_rule_pre | 55.0 (94%) | 31.2 (88%) | 61.2 (4%) | 49.2 (62%) |
| cascade | 54.9 (91%) | 30.9 (83%) | 61.8 (2%) | 49.2 (59%) |
| error_tuned | 54.8 (100%) | 31.6 (99%) | 61.8 (3%) | 49.4 (67%) |
| error_full_tuned | 54.9 (99%) | 31.6 (100%) | 61.8 (2%) | 49.4 (67%) |
| benefit_full_tuned | 54.8 (100%) | 31.7 (99%) | 61.9 (1%) | 49.5 (67%) |
| oracle | 60.1 (34%) | 35.4 (16%) | 65.9 (4%) | 53.8 (18%) |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| benefit_rule_post - error_rule | +1.80 | +1.42 | +2.18 |
| benefit_rule_post - always | +5.84 | +5.35 | +6.31 |
| benefit_rule_post - never | +13.49 | +12.40 | +14.50 |
| benefit_rule_full - error_rule | +1.45 | +1.04 | +1.87 |
| benefit_rule_full - always | +5.49 | +4.96 | +6.00 |
| benefit_rule_full - never | +13.14 | +12.06 | +14.13 |
| benefit_rule_pre - error_rule | +1.56 | +1.13 | +1.97 |
| benefit_rule_pre - always | +5.59 | +5.06 | +6.11 |
| benefit_rule_pre - never | +13.25 | +12.17 | +14.24 |
| benefit_full_tuned - error_tuned | +0.08 | +0.01 | +0.16 |
| benefit_rule_full - error_tuned | -0.34 | -0.60 | -0.11 |
| benefit_rule_full - error_rule_full | +1.81 | +1.43 | +2.20 |
| benefit_rule_full - router | -0.41 | -0.66 | -0.16 |
| benefit_rule_full - fixed_by_dataset | -0.41 | -0.66 | -0.16 |
| benefit_rule_pre - router | -0.30 | -0.51 | -0.10 |
| benefit_rule_pre - fixed_by_dataset | -0.30 | -0.51 | -0.10 |
| cascade - router | -0.24 | -0.47 | -0.02 |
| cascade - fixed_by_dataset | -0.24 | -0.47 | -0.02 |
| cascade - always | +5.65 | +5.11 | +6.17 |
| cascade - error_rule_full | +1.97 | +1.57 | +2.38 |
| benefit_full_tuned - error_full_tuned | +0.05 | -0.02 | +0.12 |

| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |
|---|---|---|---|
| evqa | +2.07 | +1.31 | +2.87 |
| infoseek | -0.05 | -0.55 | +0.42 |
| okvqa | +3.41 | +2.62 | +4.20 |

Search rates (macro): benefit_rule_pre 62%, cascade 62%, router 67%

## Mixed traffic (one gate fit on the pooled dev splits, no dataset identity)

| rule | evqa acc (reads) | infoseek acc (reads) | okvqa acc (reads) | macro acc (reads) |
|---|---|---|---|---|
| never | 26.5 (0%) | 19.3 (0%) | 61.9 (0%) | 35.9 (0%) |
| always | 54.8 (100%) | 31.6 (100%) | 44.3 (100%) | 43.6 (100%) |
| fixed_by_dataset | 54.8 (100%) | 31.6 (100%) | 61.9 (0%) | 49.5 (67%) |
| router | 54.4 (98%) | 31.5 (99%) | 61.6 (3%) | 49.1 (67%) |
| error_rule | 53.1 (79%) | 29.9 (82%) | 55.2 (42%) | 46.1 (68%) |
| error_rule_post_only | 53.1 (79%) | 29.9 (82%) | 55.2 (42%) | 46.1 (68%) |
| error_rule_full | 53.5 (83%) | 31.1 (91%) | 57.4 (32%) | 47.3 (69%) |
| benefit_rule_post | 51.9 (71%) | 30.3 (84%) | 57.4 (32%) | 46.5 (62%) |
| benefit_rule_full | 54.2 (88%) | 31.3 (91%) | 60.5 (13%) | 48.7 (64%) |
| benefit_rule_pre | 54.2 (92%) | 31.1 (92%) | 60.2 (13%) | 48.5 (66%) |
| cascade | 54.0 (87%) | 31.0 (89%) | 60.7 (10%) | 48.5 (62%) |
| error_tuned | 49.6 (66%) | 28.5 (68%) | 59.1 (23%) | 45.7 (52%) |
| error_full_tuned | 53.3 (82%) | 31.0 (90%) | 57.7 (29%) | 47.4 (67%) |
| benefit_full_tuned | 54.7 (95%) | 31.5 (95%) | 59.4 (22%) | 48.5 (70%) |
| oracle | 60.1 (34%) | 35.4 (16%) | 65.9 (4%) | 53.8 (18%) |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| benefit_rule_post - error_rule | +0.45 | +0.06 | +0.80 |
| benefit_rule_post - always | +2.97 | +2.39 | +3.49 |
| benefit_rule_post - never | +10.62 | +9.53 | +11.65 |
| benefit_rule_full - error_rule | +2.58 | +2.08 | +3.05 |
| benefit_rule_full - always | +5.09 | +4.55 | +5.58 |
| benefit_rule_full - never | +12.74 | +11.68 | +13.77 |
| benefit_rule_pre - error_rule | +2.44 | +1.92 | +2.93 |
| benefit_rule_pre - always | +4.95 | +4.43 | +5.45 |
| benefit_rule_pre - never | +12.61 | +11.55 | +13.62 |
| benefit_full_tuned - error_tuned | +2.80 | +2.30 | +3.29 |
| benefit_rule_full - error_tuned | +2.93 | +2.43 | +3.41 |
| benefit_rule_full - error_rule_full | +1.34 | +0.93 | +1.73 |
| benefit_rule_full - router | -0.49 | -0.76 | -0.22 |
| benefit_rule_full - fixed_by_dataset | -0.80 | -1.09 | -0.52 |
| benefit_rule_pre - router | -0.62 | -0.89 | -0.35 |
| benefit_rule_pre - fixed_by_dataset | -0.94 | -1.21 | -0.65 |
| cascade - router | -0.60 | -0.88 | -0.30 |
| cascade - fixed_by_dataset | -0.91 | -1.22 | -0.61 |
| cascade - always | +4.98 | +4.43 | +5.50 |
| cascade - error_rule_full | +1.22 | +0.81 | +1.62 |
| benefit_full_tuned - error_full_tuned | +1.18 | +0.80 | +1.58 |

| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |
|---|---|---|---|
| evqa | +0.73 | -0.04 | +1.57 |
| infoseek | +0.19 | -0.25 | +0.65 |
| okvqa | +3.09 | +2.25 | +3.93 |

Search rates (macro): benefit_rule_pre 66%, cascade 66%, router 67%
Router: dataset predicted from the closed-book hidden state with 92.2% accuracy.

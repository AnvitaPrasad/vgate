# Operating points (PREREG v2e, exploratory): Qwen2.5-VL-3B-Instruct, labels=official
Held-out accuracy (share of queries that read evidence). Rules are budget-free; nothing is tuned on held-out data.

## Per-dataset gates (fit on each dataset's dev split)

| rule | evqa acc (reads) | infoseek acc (reads) | okvqa acc (reads) | macro acc (reads) |
|---|---|---|---|---|
| never | 22.3 (0%) | 17.0 (0%) | 59.5 (0%) | 32.9 (0%) |
| always | 48.6 (100%) | 26.6 (100%) | 45.9 (100%) | 40.4 (100%) |
| fixed_by_dataset | 48.6 (100%) | 26.6 (100%) | 59.5 (0%) | 44.9 (67%) |
| router | 48.6 (100%) | 26.6 (100%) | 59.5 (0%) | 44.9 (67%) |
| error_rule | 47.0 (88%) | 26.4 (96%) | 57.4 (26%) | 43.6 (70%) |
| error_rule_post_only | 47.0 (88%) | 26.4 (96%) | 57.4 (26%) | 43.6 (70%) |
| error_rule_full | 47.3 (85%) | 26.1 (90%) | 57.3 (30%) | 43.6 (68%) |
| benefit_rule_post | 48.8 (95%) | 26.6 (98%) | 59.6 (6%) | 45.0 (66%) |
| benefit_rule_full | 48.3 (87%) | 26.3 (81%) | 58.3 (15%) | 44.3 (61%) |
| benefit_rule_pre | 48.7 (92%) | 26.2 (83%) | 58.5 (10%) | 44.5 (62%) |
| cascade | 48.2 (85%) | 26.0 (79%) | 58.8 (7%) | 44.4 (57%) |
| error_tuned | 48.6 (99%) | 26.5 (100%) | 59.6 (5%) | 44.9 (68%) |
| error_full_tuned | 48.7 (95%) | 26.6 (100%) | 59.6 (7%) | 45.0 (67%) |
| benefit_full_tuned | 48.7 (100%) | 26.6 (99%) | 59.3 (2%) | 44.9 (67%) |
| oracle | 53.3 (31%) | 31.0 (14%) | 65.2 (6%) | 49.8 (17%) |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| benefit_rule_post - error_rule | +1.41 | +1.05 | +1.76 |
| benefit_rule_post - always | +4.66 | +4.15 | +5.14 |
| benefit_rule_post - never | +12.07 | +11.05 | +13.09 |
| benefit_rule_full - error_rule | +0.71 | +0.26 | +1.13 |
| benefit_rule_full - always | +3.95 | +3.44 | +4.48 |
| benefit_rule_full - never | +11.37 | +10.36 | +12.36 |
| benefit_rule_pre - error_rule | +0.86 | +0.41 | +1.30 |
| benefit_rule_pre - always | +4.10 | +3.59 | +4.61 |
| benefit_rule_pre - never | +11.52 | +10.50 | +12.49 |
| benefit_full_tuned - error_tuned | -0.04 | -0.18 | +0.10 |
| benefit_rule_full - error_tuned | -0.62 | -0.95 | -0.29 |
| benefit_rule_full - error_rule_full | +0.73 | +0.31 | +1.11 |
| benefit_rule_full - router | -0.59 | -0.94 | -0.27 |
| benefit_rule_full - fixed_by_dataset | -0.59 | -0.94 | -0.27 |
| benefit_rule_pre - router | -0.44 | -0.73 | -0.15 |
| benefit_rule_pre - fixed_by_dataset | -0.44 | -0.73 | -0.15 |
| cascade - router | -0.54 | -0.88 | -0.23 |
| cascade - fixed_by_dataset | -0.54 | -0.88 | -0.23 |
| cascade - always | +4.01 | +3.45 | +4.55 |
| cascade - error_rule_full | +0.78 | +0.33 | +1.18 |
| benefit_full_tuned - error_full_tuned | -0.08 | -0.25 | +0.08 |

| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |
|---|---|---|---|
| evqa | +0.98 | +0.31 | +1.64 |
| infoseek | +0.22 | -0.38 | +0.83 |
| okvqa | +0.99 | +0.22 | +1.76 |

Search rates (macro): benefit_rule_pre 62%, cascade 62%, router 67%

## Mixed traffic (one gate fit on the pooled dev splits, no dataset identity)

| rule | evqa acc (reads) | infoseek acc (reads) | okvqa acc (reads) | macro acc (reads) |
|---|---|---|---|---|
| never | 22.3 (0%) | 17.0 (0%) | 59.5 (0%) | 32.9 (0%) |
| always | 48.6 (100%) | 26.6 (100%) | 45.9 (100%) | 40.4 (100%) |
| fixed_by_dataset | 48.6 (100%) | 26.6 (100%) | 59.5 (0%) | 44.9 (67%) |
| router | 48.2 (98%) | 26.4 (98%) | 58.9 (4%) | 44.5 (66%) |
| error_rule | 47.6 (87%) | 26.1 (85%) | 54.1 (45%) | 42.6 (72%) |
| error_rule_post_only | 47.6 (87%) | 26.1 (85%) | 54.1 (45%) | 42.6 (72%) |
| error_rule_full | 48.1 (88%) | 26.1 (89%) | 55.3 (35%) | 43.2 (71%) |
| benefit_rule_post | 46.7 (75%) | 26.1 (87%) | 55.4 (36%) | 42.7 (66%) |
| benefit_rule_full | 47.9 (85%) | 25.9 (88%) | 58.4 (16%) | 44.1 (63%) |
| benefit_rule_pre | 48.3 (90%) | 26.0 (90%) | 57.8 (16%) | 44.0 (65%) |
| cascade | 47.8 (84%) | 25.6 (85%) | 58.5 (12%) | 43.9 (60%) |
| error_tuned | 46.5 (80%) | 25.8 (78%) | 55.3 (35%) | 42.6 (65%) |
| error_full_tuned | 47.3 (83%) | 25.7 (86%) | 56.4 (26%) | 43.1 (65%) |
| benefit_full_tuned | 48.0 (85%) | 25.9 (88%) | 58.4 (16%) | 44.1 (63%) |
| oracle | 53.3 (31%) | 31.0 (14%) | 65.2 (6%) | 49.8 (17%) |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| benefit_rule_post - error_rule | +0.08 | -0.31 | +0.46 |
| benefit_rule_post - always | +2.36 | +1.86 | +2.87 |
| benefit_rule_post - never | +9.77 | +8.77 | +10.73 |
| benefit_rule_full - error_rule | +1.42 | +0.95 | +1.92 |
| benefit_rule_full - always | +3.70 | +3.20 | +4.24 |
| benefit_rule_full - never | +11.12 | +10.08 | +12.13 |
| benefit_rule_pre - error_rule | +1.40 | +0.93 | +1.88 |
| benefit_rule_pre - always | +3.68 | +3.17 | +4.20 |
| benefit_rule_pre - never | +11.10 | +10.09 | +12.10 |
| benefit_full_tuned - error_tuned | +1.50 | +1.01 | +2.00 |
| benefit_rule_full - error_tuned | +1.47 | +0.99 | +1.96 |
| benefit_rule_full - error_rule_full | +0.85 | +0.43 | +1.29 |
| benefit_rule_full - router | -0.45 | -0.78 | -0.12 |
| benefit_rule_full - fixed_by_dataset | -0.85 | -1.18 | -0.51 |
| benefit_rule_pre - router | -0.47 | -0.76 | -0.18 |
| benefit_rule_pre - fixed_by_dataset | -0.87 | -1.17 | -0.57 |
| cascade - router | -0.56 | -0.89 | -0.22 |
| cascade - fixed_by_dataset | -0.95 | -1.31 | -0.62 |
| cascade - always | +3.59 | +3.05 | +4.16 |
| cascade - error_rule_full | +0.74 | +0.32 | +1.19 |
| benefit_full_tuned - error_full_tuned | +0.98 | +0.55 | +1.40 |

| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |
|---|---|---|---|
| evqa | -0.22 | -1.01 | +0.52 |
| infoseek | -0.27 | -0.73 | +0.19 |
| okvqa | +3.04 | +2.10 | +3.98 |

Search rates (macro): benefit_rule_pre 65%, cascade 65%, router 66%
Router: dataset predicted from the closed-book hidden state with 92.6% accuracy.

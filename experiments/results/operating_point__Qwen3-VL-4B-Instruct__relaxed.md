# Operating points (PREREG v2e, exploratory): Qwen3-VL-4B-Instruct, labels=relaxed
Held-out accuracy (share of queries that read evidence). Rules are budget-free; nothing is tuned on held-out data.

## Per-dataset gates (fit on each dataset's dev split)

| rule | evqa acc (reads) | infoseek acc (reads) | okvqa acc (reads) | macro acc (reads) |
|---|---|---|---|---|
| never | 23.1 (0%) | 17.8 (0%) | 72.2 (0%) | 37.7 (0%) |
| always | 51.2 (100%) | 27.4 (100%) | 58.1 (100%) | 45.5 (100%) |
| fixed_by_dataset | 51.2 (100%) | 27.4 (100%) | 72.2 (0%) | 50.2 (67%) |
| router | 51.2 (100%) | 27.4 (100%) | 72.2 (0%) | 50.2 (67%) |
| error_rule | 49.2 (87%) | 27.3 (99%) | 71.1 (14%) | 49.2 (67%) |
| error_rule_post_only | 49.2 (87%) | 27.3 (99%) | 71.1 (14%) | 49.2 (67%) |
| error_rule_full | 49.9 (85%) | 27.1 (87%) | 70.8 (17%) | 49.3 (63%) |
| benefit_rule_post | 51.2 (98%) | 26.8 (94%) | 72.1 (1%) | 50.0 (65%) |
| benefit_rule_full | 50.8 (90%) | 27.8 (77%) | 72.0 (12%) | 50.2 (60%) |
| benefit_rule_pre | 50.8 (95%) | 28.0 (81%) | 71.8 (7%) | 50.2 (61%) |
| cascade | 50.7 (89%) | 27.6 (74%) | 72.1 (6%) | 50.1 (57%) |
| error_tuned | 51.1 (100%) | 27.4 (100%) | 72.1 (1%) | 50.2 (67%) |
| error_full_tuned | 51.2 (100%) | 27.2 (96%) | 72.1 (0%) | 50.2 (66%) |
| benefit_full_tuned | 51.2 (100%) | 27.8 (82%) | 72.1 (0%) | 50.4 (61%) |
| oracle | 55.9 (33%) | 31.9 (14%) | 78.2 (6%) | 55.4 (18%) |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| benefit_rule_post - error_rule | +0.84 | +0.51 | +1.16 |
| benefit_rule_post - always | +4.50 | +3.97 | +5.01 |
| benefit_rule_post - never | +12.38 | +11.28 | +13.34 |
| benefit_rule_full - error_rule | +0.99 | +0.60 | +1.37 |
| benefit_rule_full - always | +4.65 | +4.13 | +5.17 |
| benefit_rule_full - never | +12.53 | +11.47 | +13.53 |
| benefit_rule_pre - error_rule | +1.03 | +0.65 | +1.41 |
| benefit_rule_pre - always | +4.69 | +4.17 | +5.21 |
| benefit_rule_pre - never | +12.56 | +11.49 | +13.55 |
| benefit_full_tuned - error_tuned | +0.19 | -0.01 | +0.40 |
| benefit_rule_full - error_tuned | -0.00 | -0.33 | +0.32 |
| benefit_rule_full - error_rule_full | +0.90 | +0.49 | +1.29 |
| benefit_rule_full - router | -0.05 | -0.37 | +0.27 |
| benefit_rule_full - fixed_by_dataset | -0.05 | -0.37 | +0.27 |
| benefit_rule_pre - router | -0.02 | -0.29 | +0.25 |
| benefit_rule_pre - fixed_by_dataset | -0.02 | -0.29 | +0.25 |
| cascade - router | -0.11 | -0.42 | +0.20 |
| cascade - fixed_by_dataset | -0.11 | -0.42 | +0.20 |
| cascade - always | +4.60 | +4.06 | +5.14 |
| cascade - error_rule_full | +0.85 | +0.45 | +1.26 |
| benefit_full_tuned - error_full_tuned | +0.21 | +0.02 | +0.41 |

| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |
|---|---|---|---|
| evqa | +0.91 | +0.19 | +1.65 |
| infoseek | +0.63 | +0.03 | +1.26 |
| okvqa | +1.16 | +0.52 | +1.83 |

Search rates (macro): benefit_rule_pre 61%, cascade 61%, router 67%

## Mixed traffic (one gate fit on the pooled dev splits, no dataset identity)

| rule | evqa acc (reads) | infoseek acc (reads) | okvqa acc (reads) | macro acc (reads) |
|---|---|---|---|---|
| never | 23.1 (0%) | 17.8 (0%) | 72.2 (0%) | 37.7 (0%) |
| always | 51.2 (100%) | 27.4 (100%) | 58.1 (100%) | 45.5 (100%) |
| fixed_by_dataset | 51.2 (100%) | 27.4 (100%) | 72.2 (0%) | 50.2 (67%) |
| router | 50.6 (98%) | 27.3 (99%) | 71.7 (4%) | 49.9 (67%) |
| error_rule | 47.9 (75%) | 26.4 (78%) | 67.3 (37%) | 47.2 (63%) |
| error_rule_post_only | 47.9 (75%) | 26.4 (78%) | 67.3 (37%) | 47.2 (63%) |
| error_rule_full | 49.6 (85%) | 27.2 (90%) | 69.8 (20%) | 48.9 (65%) |
| benefit_rule_post | 47.1 (68%) | 26.9 (82%) | 68.5 (34%) | 47.5 (62%) |
| benefit_rule_full | 49.6 (86%) | 27.6 (84%) | 71.0 (13%) | 49.4 (61%) |
| benefit_rule_pre | 50.3 (92%) | 27.4 (85%) | 71.0 (13%) | 49.6 (63%) |
| cascade | 49.5 (85%) | 27.2 (80%) | 71.7 (9%) | 49.4 (58%) |
| error_tuned | 46.0 (66%) | 25.9 (68%) | 69.0 (26%) | 47.0 (53%) |
| error_full_tuned | 50.8 (90%) | 27.3 (94%) | 67.8 (30%) | 48.6 (71%) |
| benefit_full_tuned | 49.3 (84%) | 27.7 (81%) | 71.2 (12%) | 49.4 (59%) |
| oracle | 55.9 (33%) | 31.9 (14%) | 78.2 (6%) | 55.4 (18%) |

| macro accuracy difference | point | 95% lo | 95% hi |
|---|---|---|---|
| benefit_rule_post - error_rule | +0.27 | -0.15 | +0.68 |
| benefit_rule_post - always | +1.95 | +1.37 | +2.47 |
| benefit_rule_post - never | +9.83 | +8.76 | +10.78 |
| benefit_rule_full - error_rule | +2.18 | +1.65 | +2.70 |
| benefit_rule_full - always | +3.85 | +3.32 | +4.39 |
| benefit_rule_full - never | +11.73 | +10.68 | +12.70 |
| benefit_rule_pre - error_rule | +2.35 | +1.81 | +2.91 |
| benefit_rule_pre - always | +4.03 | +3.51 | +4.56 |
| benefit_rule_pre - never | +11.90 | +10.86 | +12.91 |
| benefit_full_tuned - error_tuned | +2.43 | +1.89 | +2.97 |
| benefit_rule_full - error_tuned | +2.42 | +1.88 | +2.96 |
| benefit_rule_full - error_rule_full | +0.50 | +0.10 | +0.90 |
| benefit_rule_full - router | -0.48 | -0.86 | -0.14 |
| benefit_rule_full - fixed_by_dataset | -0.85 | -1.24 | -0.50 |
| benefit_rule_pre - router | -0.30 | -0.61 | +0.02 |
| benefit_rule_pre - fixed_by_dataset | -0.68 | -1.01 | -0.35 |
| cascade - router | -0.43 | -0.81 | -0.08 |
| cascade - fixed_by_dataset | -0.80 | -1.19 | -0.44 |
| cascade - always | +3.90 | +3.33 | +4.47 |
| cascade - error_rule_full | +0.55 | +0.13 | +0.99 |
| benefit_full_tuned - error_full_tuned | +0.78 | +0.35 | +1.19 |

| VGate-full rule − error rule (same features), by dataset | point | 95% lo | 95% hi |
|---|---|---|---|
| evqa | -0.04 | -0.83 | +0.76 |
| infoseek | +0.38 | -0.19 | +0.94 |
| okvqa | +1.16 | +0.44 | +1.90 |

Search rates (macro): benefit_rule_pre 63%, cascade 63%, router 67%
Router: dataset predicted from the closed-book hidden state with 91.1% accuracy.

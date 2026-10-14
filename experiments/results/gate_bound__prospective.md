# The gate estimate and ceiling on runs made after they were registered (PREREG v9, v9a)

## B1 — ReflectiVA checkpoint, [Retrieval] token (J17)

- evqa: {"n": 2750, "help": 32.9, "harm": 4.7, "auroc": 0.509, "needed": 0.881, "floor": 0.606, "lead_at_published_decision": 0.0, "estimate": 0.0, "ceiling": 0.08, "best_lead": 0.0, "B1_at_most_estimate_plus_0p3": true, "B1_below_one_point_where_auroc_below_needed": true, "B4_best_lead_at_most_ceiling": true}
- infoseek: {"n": 3637, "help": 14.2, "harm": 4.6, "auroc": 0.47, "needed": 0.823, "floor": 0.609, "lead_at_published_decision": 0.0, "estimate": 0.0, "ceiling": 0.0, "best_lead": 0.05, "B1_at_most_estimate_plus_0p3": true, "B1_below_one_point_where_auroc_below_needed": true, "B4_best_lead_at_most_ceiling": false}
- okvqa: {"n": 4046, "help": 4.9, "harm": 24.5, "auroc": 0.525, "needed": 0.857, "floor": 0.602, "lead_at_published_decision": -16.21, "estimate": 0.0, "ceiling": 0.25, "best_lead": 0.05, "B1_at_most_estimate_plus_0p3": true, "B1_below_one_point_where_auroc_below_needed": true, "B4_best_lead_at_most_ceiling": true}

## B2, B3 — Qwen2.5-VL-3B-Instruct on CVQA (held-out, n = 8373; helps 10.2%, harms 14.0%; AUROC needed 0.643)

- benefit_full: {"auroc": 0.619, "best_lead": 1.23, "estimate": 0.66, "ceiling": 2.42, "registered_rule_lead": 0.85, "B2_best_lead_within_0p5_of_estimate": false, "B2_rule_at_most_estimate_plus_0p3": true, "B3_prediction": "does not", "B3_best_lead_at_least_one_point": true, "B4_best_lead_at_most_ceiling": true, "B3_correct": false}
- benefit_post: {"auroc": 0.611, "best_lead": 0.82, "estimate": 0.56, "ceiling": 2.26, "registered_rule_lead": 0.76, "B2_best_lead_within_0p5_of_estimate": true, "B2_rule_at_most_estimate_plus_0p3": true, "B3_prediction": "does not", "B3_best_lead_at_least_one_point": false, "B4_best_lead_at_most_ceiling": true, "B3_correct": true}
- after_reading: {"auroc": 0.677, "best_lead": 1.91, "estimate": 1.51, "ceiling": 3.6, "registered_rule_lead": 0.87, "B2_best_lead_within_0p5_of_estimate": true, "B2_rule_at_most_estimate_plus_0p3": true, "B3_prediction": "leads by one point", "B3_best_lead_at_least_one_point": true, "B4_best_lead_at_most_ceiling": true, "B3_correct": true}

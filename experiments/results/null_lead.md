# The planner's noise floor (PREREG v12; exploratory)

Null lead: what the best threshold of a score with no signal leads the better fixed policy by (points), exactly; with sign: the direction of the score chosen too (4,000 shuffles).

Case study, 156 held-out scores in 12 cells: the null lead is at most 0.013 points (0.022 with sign; 0.015 on average over the scores). The scores' best leads average 0.10 points and their split-half leads -0.02; 90 of 156 best leads are above their cell's null lead with sign.

| reader | benchmark | n | helps % | harms % | null lead | with sign | mean best lead of its scores | mean split-half lead |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | evqa | 2750 | 32.9 | 4.7 | 0.006 | 0.011 | 0.042 | -0.077 |
| Qwen3-VL-4B-Instruct | infoseek | 3637 | 14.2 | 4.6 | 0.013 | 0.022 | 0.221 | 0.124 |
| Qwen3-VL-4B-Instruct | okvqa | 4046 | 4.9 | 24.5 | 0.006 | 0.012 | 0.097 | 0.042 |
| Qwen2.5-VL-3B-Instruct | evqa | 2750 | 31.0 | 4.7 | 0.006 | 0.012 | 0.107 | -0.082 |
| Qwen2.5-VL-3B-Instruct | infoseek | 3637 | 14.0 | 4.4 | 0.013 | 0.022 | 0.076 | -0.034 |
| Qwen2.5-VL-3B-Instruct | okvqa | 4046 | 5.6 | 19.3 | 0.01 | 0.019 | 0.075 | -0.006 |
| InternVL3_5-4B-HF | evqa | 2750 | 37.8 | 4.1 | 0.004 | 0.008 | 0.043 | -0.072 |
| InternVL3_5-4B-HF | infoseek | 3637 | 13.7 | 3.5 | 0.009 | 0.017 | 0.082 | -0.063 |
| InternVL3_5-4B-HF | okvqa | 4046 | 6.1 | 23.8 | 0.008 | 0.015 | 0.299 | 0.159 |
| Qwen3-VL-8B-Instruct | evqa | 2750 | 33.6 | 5.3 | 0.007 | 0.013 | 0.104 | -0.079 |
| Qwen3-VL-8B-Instruct | infoseek | 3637 | 16.1 | 3.8 | 0.008 | 0.015 | 0.025 | -0.087 |
| Qwen3-VL-8B-Instruct | okvqa | 4046 | 4.0 | 21.7 | 0.006 | 0.01 | 0.023 | -0.012 |

| log | n | helps % | harms % | oracle lead | null lead | with sign |
|---|---|---|---|---|---|---|
| published log (GPT-4o, six sources pooled) | 600 | 7.5 | 6.8 | 6.8 | 0.63 | 0.97 |
| published log, source life | 100 | 13.0 | 5.0 | 5.0 | 0.47 | 0.83 |
| published log, source in-house 2 | 100 | 5.0 | 10.0 | 5.0 | 0.65 | 1.15 |
| published log, source dyn-vqa zh | 100 | 18.0 | 13.0 | 13.0 | 1.46 | 2.42 |
| published log, source dyn-vqa en | 100 | 5.0 | 1.0 | 1.0 | 0.17 | 0.33 |
| published log, source nocaps | 100 | 2.0 | 2.0 | 2.0 | 0.83 | 1.33 |
| published log, source visual7w | 100 | 2.0 | 10.0 | 2.0 | 0.2 | 0.38 |
| reference: helps 10%, harms 10%, 1,000 queries | 1000 | 10.0 | 10.0 | 10.0 | 0.84 | 1.18 |
| reference: helps 10%, harms 10%, 10,000 queries | 10000 | 10.0 | 10.0 | 10.0 | 0.28 | 0.38 |

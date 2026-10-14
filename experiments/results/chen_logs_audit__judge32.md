# PREREG v11b — the audit on Chen et al.'s released logs under a judge model's score (Mix set, 600 questions from six sources, %)

Told the source: the better of never / always per source, chosen on four folds and applied to the fifth. Differences to it with 95% bootstrap intervals over questions.

| reader | never | always | hard gate | soft gate | prompt | human | told the source (in sample) | oracle | hard − told | soft − told | human − told | always − told |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4o | 41.18 | 44.82 | 44.93 | 44.57 | 40.79 | 45.93 | 45.74 (46.16) | 53.02 | -0.81 [-2.30, +0.63] | -1.18 [-2.86, +0.47] | +0.19 [-1.57, +1.94] | -0.92 [-2.43, +0.51] |
| Qwen-VL-Max | 39.3 | 43.83 | 44.94 | 43.38 | 40.6 | 44.34 | 45.9 (45.9) | 53.1 | -0.96 [-1.90, -0.02] | -2.52 [-4.11, -0.90] | -1.56 [-3.19, +0.03] | -2.07 [-3.40, -0.82] |
| Qwen-VL-2 | 39.45 | 43.54 | 44.46 | 43.31 | 40.86 | 44.35 | 45.31 (45.61) | 52.87 | -0.86 [-1.85, +0.13] | -2.00 [-3.52, -0.47] | -0.96 [-2.62, +0.67] | -1.77 [-3.17, -0.40] |
| Qwen-VL-Chat | 28.55 | 31.79 | 32.83 | 33.2 | 29.03 | 31.95 | 35.07 (35.3) | 40.84 | -2.24 [-3.50, -0.98] | -1.87 [-3.22, -0.64] | -3.12 [-4.85, -1.22] | -3.28 [-4.74, -1.78] |
| DeepSeek-VL-Chat | 28.4 | 40.65 | 39.88 | 37.87 | 29.35 | 36.31 | 41.49 (41.49) | 45.78 | -1.61 [-2.70, -0.51] | -3.62 [-5.38, -1.95] | -5.18 [-7.20, -3.20] | -0.84 [-1.50, -0.23] |

Reproduction of their Tables 3 and 11 (Mix, Acc.; never, always, prompt, hard, soft, human): GPT-4o largest difference 10.26; Qwen-VL-Max largest difference 7.71; Qwen-VL-2 largest difference 7.06; Qwen-VL-Chat largest difference 6.81; DeepSeek-VL-Chat largest difference 6.56.
The hard gate's decisions equal its own majority decision per source on 82.8% of the questions; search rates (hard / soft), by source: life 96% / 59%; in-house 2 100% / 81%; dyn-vqa zh 94% / 77%; dyn-vqa en 89% / 71%; nocaps 42% / 0%; visual7w 40% / 8%.
Majority decision per source of the hard / soft gate, difference to told-the-source: GPT-4o -0.11 [-1.28, +1.03] / -0.11 [-1.28, +1.03]; Qwen-VL-Max +0.00 [+0.00, +0.00] / +0.00 [+0.00, +0.00]; Qwen-VL-2 +0.30 [+0.00, +0.68] / +0.30 [+0.00, +0.68]; Qwen-VL-Chat -0.38 [-1.16, +0.41] / -0.38 [-1.16, +0.41]; DeepSeek-VL-Chat -0.67 [-1.68, +0.40] / -0.67 [-1.68, +0.40].
200 other fold assignments (hard gate − told on average, share of assignments in which the hard gate is ahead; the same for the soft gate): GPT-4o -0.42, 8.0%; -0.79, 0.5%; Qwen-VL-Max -0.68, 7.0%; -2.24, 0.0%; Qwen-VL-2 -0.59, 3.5%; -1.73, 0.0%; Qwen-VL-Chat -2.23, 0.0%; -1.86, 0.0%; DeepSeek-VL-Chat -1.58, 0.0%; -3.60, 0.0%.
"Human" is their boundary model trained on human-labelled data, not a person deciding. The accuracy is their token accuracy; their LLM-judge score (a paid API) is not computed, and under it their Table 4 prints the hard gate above always-search for GPT-4o (54.10 against 52.90).
Told-the-source under five other fold assignments: GPT-4o 45.34–45.65; Qwen-VL-Max 45.36–45.9; Qwen-VL-2 44.76–45.41; Qwen-VL-Chat 34.94–35.3; DeepSeek-VL-Chat 40.81–41.49.

| reader | never / always by source (life, in-house 2, dyn-vqa zh, dyn-vqa en, nocaps, visual7w) | helps % | harms % | AUROC floor | needed |
|---|---|---|---|---|---|
| GPT-4o | 31.4 / 42.0; 27.7 / 24.6; 28.5 / 47.6; 54.5 / 54.7; 67.7 / 67.7; 37.3 / 32.4 | 15.3 | 10.0 | 0.55 | 0.663 |
| Qwen-VL-Max | 33.2 / 53.0; 24.9 / 28.7; 24.9 / 35.8; 39.8 / 44.8; 67.8 / 57.6; 45.2 / 43.0 | 15.5 | 12.5 | 0.54 | 0.61 |
| Qwen-VL-2 | 31.8 / 49.9; 26.9 / 28.1; 25.3 / 37.2; 39.3 / 45.0; 68.2 / 58.0; 45.2 / 43.0 | 14.8 | 11.3 | 0.544 | 0.626 |
| Qwen-VL-Chat | 20.6 / 31.3; 17.1 / 13.9; 13.8 / 26.9; 21.0 / 37.3; 55.0 / 37.1; 43.8 / 44.3 | 13.8 | 11.2 | 0.545 | 0.616 |
| DeepSeek-VL-Chat | 18.7 / 46.1; 13.9 / 20.9; 13.0 / 31.4; 17.1 / 38.9; 67.8 / 62.7; 39.9 / 43.9 | 20.5 | 5.5 | 0.591 | 0.822 |

Expectations: K1 held; K2 held; K3 NOT held; K4 held.

Judge against their printed scores (never, always; points): GPT-4o [10.26, 8.08]; Qwen-VL-Max [7.24, 5.43]; Qwen-VL-2 [6.91, 4.35]; Qwen-VL-Chat [5.89, 6.81]; DeepSeek-VL-Chat [6.56, 4.53]. Validity gate (within 5.0 points for at least four readers): NOT passed.
Expectations: L1 held; L2 held; L3 held.

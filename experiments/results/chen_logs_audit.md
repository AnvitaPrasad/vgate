# PREREG v11 — the audit on Chen et al.'s released logs (Mix set, 600 questions from six sources; their graded token accuracy, %)

Told the source: the better of never / always per source, chosen on four folds and applied to the fifth. Differences to it with 95% bootstrap intervals over questions.

| reader | never | always | hard gate | soft gate | prompt | human | told the source (in sample) | oracle | hard − told | soft − told | human − told | always − told |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4o | 34.52 | 35.96 | 35.9 | 35.86 | 33.99 | 36.49 | 38.28 (38.28) | 42.73 | -2.38 [-3.92, -0.86] | -2.42 [-4.01, -0.76] | -1.80 [-3.26, -0.09] | -2.32 [-4.03, -0.71] |
| Qwen-VL-Max | 32.35 | 34.78 | 35.48 | 34.84 | 33.0 | 35.51 | 36.35 (36.99) | 42.83 | -0.87 [-2.23, +0.52] | -1.50 [-3.34, +0.32] | -0.84 [-2.57, +0.92] | -1.57 [-3.09, -0.06] |
| Qwen-VL-2 | 32.45 | 35.56 | 36.29 | 35.62 | 33.13 | 36.33 | 36.67 (37.44) | 42.72 | -0.38 [-1.66, +1.02] | -1.04 [-2.61, +0.60] | -0.34 [-1.76, +1.13] | -1.11 [-2.56, +0.42] |
| Qwen-VL-Chat | 26.13 | 32.39 | 32.73 | 30.98 | 27.23 | 31.02 | 33.59 (34.05) | 36.96 | -0.86 [-1.76, +0.06] | -2.61 [-4.22, -1.07] | -2.58 [-4.28, -0.76] | -1.20 [-2.10, -0.30] |
| DeepSeek-VL-Chat | 25.81 | 35.23 | 35.38 | 33.06 | 26.55 | 32.73 | 35.73 (36.09) | 39.86 | -0.35 [-1.48, +0.74] | -2.67 [-4.30, -1.12] | -3.00 [-4.73, -1.31] | -0.50 [-1.40, +0.37] |

Reproduction of their Tables 3 and 11 (Mix, Acc.; never, always, prompt, hard, soft, human): GPT-4o largest difference 0.0; Qwen-VL-Max largest difference 0.0; Qwen-VL-2 largest difference 0.14; Qwen-VL-Chat largest difference 0.0; DeepSeek-VL-Chat largest difference 0.0.
The hard gate's decisions equal its own majority decision per source on 82.8% of the questions; search rates (hard / soft), by source: life 96% / 59%; in-house 2 100% / 81%; dyn-vqa zh 94% / 77%; dyn-vqa en 89% / 71%; nocaps 42% / 0%; visual7w 40% / 8%.
Majority decision per source of the hard / soft gate, difference to told-the-source: GPT-4o -1.68 [-2.98, -0.37] / -1.68 [-2.98, -0.37]; Qwen-VL-Max +0.14 [-0.78, +1.17] / +0.14 [-0.78, +1.17]; Qwen-VL-2 +0.77 [-0.07, +1.74] / +0.77 [-0.07, +1.74]; Qwen-VL-Chat -0.35 [-1.26, +0.56] / -0.35 [-1.26, +0.56]; DeepSeek-VL-Chat +0.36 [-0.22, +1.06] / +0.36 [-0.22, +1.06].
200 other fold assignments (hard gate − told on average, share of assignments in which the hard gate is ahead; the same for the soft gate): GPT-4o -2.37, 0.0%; -2.40, 0.0%; Qwen-VL-Max -1.00, 1.5%; -1.63, 0.0%; Qwen-VL-2 -0.36, 5.0%; -1.03, 0.0%; Qwen-VL-Chat -1.05, 0.0%; -2.80, 0.0%; DeepSeek-VL-Chat -0.48, 8.5%; -2.80, 0.0%.
"Human" is their boundary model trained on human-labelled data, not a person deciding. The accuracy is their token accuracy; their LLM-judge score (a paid API) is not computed, and under it their Table 4 prints the hard gate above always-search for GPT-4o (54.10 against 52.90).
Told-the-source under five other fold assignments: GPT-4o 38.28–38.28; Qwen-VL-Max 36.17–36.54; Qwen-VL-2 36.41–36.83; Qwen-VL-Chat 33.66–34.05; DeepSeek-VL-Chat 35.55–36.09.

| reader | never / always by source (life, in-house 2, dyn-vqa zh, dyn-vqa en, nocaps, visual7w) | helps % | harms % | AUROC floor | needed |
|---|---|---|---|---|---|
| GPT-4o | 32.0 / 40.8; 33.8 / 27.8; 46.0 / 52.2; 24.3 / 27.8; 32.5 / 36.6; 38.6 / 30.7 | 7.5 | 6.8 | 0.573 | 0.628 |
| Qwen-VL-Max | 26.8 / 45.0; 29.3 / 26.3; 36.2 / 44.5; 20.9 / 22.3; 34.8 / 32.9; 46.1 / 37.8 | 11.0 | 7.3 | 0.568 | 0.684 |
| Qwen-VL-2 | 25.8 / 44.8; 30.0 / 30.4; 36.2 / 46.0; 21.6 / 22.3; 35.3 / 32.4; 45.9 / 37.4 | 10.7 | 6.8 | 0.573 | 0.697 |
| Qwen-VL-Chat | 18.1 / 42.5; 17.3 / 16.4; 27.0 / 38.2; 12.7 / 20.7; 38.9 / 29.8; 42.9 / 46.8 | 11.5 | 5.3 | 0.594 | 0.768 |
| DeepSeek-VL-Chat | 13.3 / 41.8; 16.3 / 20.6; 21.8 / 46.3; 17.6 / 22.0; 42.8 / 39.7; 43.1 / 41.1 | 15.3 | 4.3 | 0.615 | 0.841 |

Expectations: K1 held; K2 held; K3 NOT held; K4 held.

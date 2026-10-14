# PREREG v11b — the audit on Chen et al.'s released logs under a judge model's score (Mix set, 600 questions from six sources, %)

Told the source: the better of never / always per source, chosen on four folds and applied to the fifth. Differences to it with 95% bootstrap intervals over questions.

| reader | never | always | hard gate | soft gate | prompt | human | told the source (in sample) | oracle | hard − told | soft − told | human − told | always − told |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-4o | 38.32 | 43.18 | 43.0 | 41.74 | 38.41 | 43.41 | 43.95 (43.95) | 51.37 | -0.95 [-1.85, -0.07] | -2.21 [-3.92, -0.53] | -0.53 [-2.42, +1.23] | -0.77 [-1.65, -0.03] |
| Qwen-VL-Max | 35.7 | 40.99 | 41.78 | 40.34 | 37.0 | 41.15 | 41.74 (42.76) | 51.19 | +0.04 [-0.97, +1.06] | -1.40 [-3.28, +0.37] | -0.59 [-2.43, +1.30] | -0.75 [-2.09, +0.60] |
| Qwen-VL-2 | 35.94 | 40.76 | 41.63 | 40.18 | 37.36 | 41.19 | 41.97 (42.32) | 50.64 | -0.34 [-1.33, +0.71] | -1.79 [-3.54, -0.07] | -0.78 [-2.66, +1.08] | -1.22 [-2.39, -0.04] |
| Qwen-VL-Chat | 26.15 | 29.41 | 30.21 | 31.16 | 26.3 | 29.97 | 32.77 (32.77) | 38.21 | -2.55 [-3.80, -1.32] | -1.61 [-2.92, -0.32] | -2.80 [-4.53, -0.92] | -3.36 [-4.73, -2.01] |
| DeepSeek-VL-Chat | 28.58 | 38.91 | 39.05 | 37.15 | 28.91 | 35.56 | 39.72 (39.73) | 45.52 | -0.67 [-1.54, +0.16] | -2.57 [-3.96, -1.27] | -4.17 [-6.21, -2.38] | -0.82 [-2.05, +0.40] |

Reproduction of their Tables 3 and 11 (Mix, Acc.; never, always, prompt, hard, soft, human): GPT-4o largest difference 13.12; Qwen-VL-Max largest difference 10.9; Qwen-VL-2 largest difference 10.42; Qwen-VL-Chat largest difference 9.38; DeepSeek-VL-Chat largest difference 6.8.
The hard gate's decisions equal its own majority decision per source on 82.8% of the questions; search rates (hard / soft), by source: life 96% / 59%; in-house 2 100% / 81%; dyn-vqa zh 94% / 77%; dyn-vqa en 89% / 71%; nocaps 42% / 0%; visual7w 40% / 8%.
Majority decision per source of the hard / soft gate, difference to told-the-source: GPT-4o -0.61 [-1.40, +0.12] / -0.61 [-1.40, +0.12]; Qwen-VL-Max +1.02 [+0.20, +1.95] / +1.02 [+0.20, +1.95]; Qwen-VL-2 +0.28 [-0.50, +1.13] / +0.28 [-0.50, +1.13]; Qwen-VL-Chat -0.68 [-1.56, +0.26] / -0.68 [-1.56, +0.26]; DeepSeek-VL-Chat +0.00 [+0.00, +0.00] / +0.00 [+0.00, +0.00].
200 other fold assignments (hard gate − told on average, share of assignments in which the hard gate is ahead; the same for the soft gate): GPT-4o -0.60, 8.0%; -1.86, 0.0%; Qwen-VL-Max -0.26, 24.0%; -1.70, 0.0%; Qwen-VL-2 +0.22, 70.5%; -1.23, 0.5%; Qwen-VL-Chat -2.45, 0.0%; -1.51, 0.0%; DeepSeek-VL-Chat -0.52, 3.5%; -2.42, 0.0%.
"Human" is their boundary model trained on human-labelled data, not a person deciding. The accuracy is their token accuracy; their LLM-judge score (a paid API) is not computed, and under it their Table 4 prints the hard gate above always-search for GPT-4o (54.10 against 52.90).
Told-the-source under five other fold assignments: GPT-4o 43.51–43.95; Qwen-VL-Max 41.36–42.46; Qwen-VL-2 40.91–41.79; Qwen-VL-Chat 32.43–32.77; DeepSeek-VL-Chat 39.07–39.72.

| reader | never / always by source (life, in-house 2, dyn-vqa zh, dyn-vqa en, nocaps, visual7w) | helps % | harms % | AUROC floor | needed |
|---|---|---|---|---|---|
| GPT-4o | 27.2 / 41.9; 23.9 / 25.8; 39.2 / 49.8; 50.3 / 53.2; 56.2 / 59.9; 33.1 / 28.5 | 15.8 | 11.0 | 0.545 | 0.646 |
| Qwen-VL-Max | 25.1 / 49.2; 22.2 / 28.8; 31.2 / 39.6; 40.3 / 43.5; 55.4 / 45.4; 40.0 / 39.4 | 17.8 | 13.5 | 0.537 | 0.618 |
| Qwen-VL-2 | 25.0 / 46.9; 23.4 / 26.4; 31.1 / 41.8; 41.3 / 43.6; 55.2 / 45.9; 39.6 / 40.0 | 17.2 | 13.3 | 0.537 | 0.614 |
| Qwen-VL-Chat | 17.2 / 29.1; 15.8 / 13.3; 20.7 / 32.9; 23.4 / 37.5; 43.3 / 25.8; 36.4 / 37.9 | 13.2 | 10.2 | 0.549 | 0.632 |
| DeepSeek-VL-Chat | 16.4 / 39.4; 15.6 / 23.6; 16.1 / 34.6; 22.5 / 39.9; 54.2 / 51.7; 46.7 / 44.3 | 18.3 | 7.5 | 0.567 | 0.752 |

Expectations: K1 NOT held; K2 held; K3 NOT held; K4 held.

Judge against their printed scores (never, always; points): GPT-4o [13.12, 9.72]; Qwen-VL-Max [10.84, 8.27]; Qwen-VL-2 [10.42, 7.13]; Qwen-VL-Chat [8.29, 9.19]; DeepSeek-VL-Chat [6.38, 6.27]. Validity gate (within 5.0 points for at least four readers): NOT passed.
Expectations: L1 held; L2 held; L3 held.

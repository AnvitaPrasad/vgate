# PREREG v14b — the audit on Chen et al.'s released logs, Mix set restricted to 400 questions from 4 sources (life, in-house 2, nocaps, visual7w); token accuracy, %

Told the source: the better of never / always per source, chosen on four folds and applied to the fifth. Differences to it with 95% bootstrap intervals over questions.

| reader | never | always | hard gate | soft gate | prompt | human | told the source (in sample) | oracle | hard − told | soft − told | human − told | always − told |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-VL-Chat | 28.87 | 35.78 | 36.39 | 35.08 | 29.93 | 35.11 | 36.05 (37.07) | 40.76 | +0.34 [-0.92, +1.66] | -0.97 [-2.83, +0.94] | -0.94 [-2.65, +0.75] | -0.27 [-1.45, +0.99] |
| GPT-4o | 34.21 | 33.96 | 34.4 | 35.33 | 33.37 | 35.33 | 37.44 (37.44) | 41.26 | -3.04 [-5.26, -0.84] | -2.11 [-4.11, -0.06] | -2.11 [-4.14, -0.14] | -3.48 [-5.96, -0.96] |
| Qwen-VL-2 | 34.22 | 36.26 | 37.07 | 37.54 | 34.68 | 37.83 | 38.46 (39.08) | 43.72 | -1.39 [-3.05, +0.18] | -0.92 [-2.93, +1.00] | -0.63 [-2.15, +0.94] | -2.20 [-4.37, -0.23] |
| Qwen-VL-Chat | 29.29 | 33.89 | 34.3 | 33.12 | 30.62 | 34.29 | 35.96 (36.37) | 39.11 | -1.66 [-3.04, -0.37] | -2.83 [-4.94, -0.92] | -1.67 [-3.58, +0.15] | -2.07 [-3.37, -0.78] |
| Qwen-VL-Max | 34.26 | 35.48 | 36.12 | 36.64 | 34.59 | 37.38 | 38.79 (38.79) | 43.8 | -2.67 [-4.91, -0.43] | -2.15 [-4.52, +0.09] | -1.41 [-3.28, +0.39] | -3.31 [-5.81, -0.92] |

The hard gate's decisions equal its own majority decision per source on 78.5% of the questions; search rates (hard / soft), by source: life 96% / 59%; in-house 2 100% / 81%; nocaps 42% / 0%; visual7w 40% / 8%.
Majority decision per source of the hard / soft gate, difference to told-the-source: DeepSeek-VL-Chat +1.02 [+0.08, +2.19] / +1.02 [+0.08, +2.19]; GPT-4o -2.53 [-4.49, -0.57] / -2.53 [-4.49, -0.57]; Qwen-VL-2 +0.62 [-0.38, +1.81] / +0.62 [-0.38, +1.81]; Qwen-VL-Chat -0.79 [-2.23, +0.60] / -0.79 [-2.23, +0.60]; Qwen-VL-Max -0.74 [-2.53, +1.21] / -0.74 [-2.53, +1.21].
200 other fold assignments (hard gate − told on average, share of assignments in which the hard gate is ahead; the same for the soft gate): DeepSeek-VL-Chat -0.35, 18.0%; -1.66, 0.0%; GPT-4o -3.04, 0.0%; -2.11, 0.0%; Qwen-VL-2 -1.31, 0.0%; -0.84, 0.5%; Qwen-VL-Chat -1.65, 0.0%; -2.83, 0.0%; Qwen-VL-Max -2.43, 0.0%; -1.91, 0.0%.
"Human" is their boundary model trained on human-labelled data, not a person deciding. The accuracy is their token accuracy; their LLM-judge score (a paid API) is not computed, and under it their Table 4 prints the hard gate above always-search for GPT-4o (54.10 against 52.90).
Told-the-source under five other fold assignments: DeepSeek-VL-Chat 35.28–37.07; GPT-4o 37.44–37.44; Qwen-VL-2 38.26–38.45; Qwen-VL-Chat 35.73–36.37; Qwen-VL-Max 37.88–38.79.

| reader | never / always by source (life, in-house 2, nocaps, visual7w) | helps % | harms % | AUROC floor | needed |
|---|---|---|---|---|---|
| DeepSeek-VL-Chat | 13.3 / 41.8; 16.3 / 20.6; 42.8 / 39.7; 43.1 / 41.1 | 12.5 | 4.5 | 0.611 | 0.814 |
| GPT-4o | 32.0 / 40.8; 33.8 / 27.8; 32.5 / 36.6; 38.6 / 30.7 | 5.5 | 6.8 | 0.591 | 0.676 |
| Qwen-VL-2 | 25.8 / 44.8; 30.0 / 30.4; 35.3 / 32.4; 45.9 / 37.4 | 9.5 | 7.5 | 0.567 | 0.651 |
| Qwen-VL-Chat | 18.1 / 42.5; 17.3 / 16.4; 38.9 / 29.8; 42.9 / 46.8 | 10.0 | 6.2 | 0.58 | 0.71 |
| Qwen-VL-Max | 26.8 / 45.0; 29.3 / 26.3; 34.8 / 32.9; 46.1 / 37.8 | 9.8 | 8.2 | 0.561 | 0.628 |

Expectations: K1 held; K2 held; K3 NOT held; K4 held.

Expectations (v14b): B1 held; B2 held.

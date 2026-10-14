# PREREG v10b — gates fitted on development data and applied to held-out data: logistic (the paper's VGate-full) and boosted trees (exploratory, not blind)

Lead of the rule "read iff P(help) - P(harm) > 0" over the held-out split's better fixed policy, points, 95% image-cluster intervals. Seen / unseen: held-out questions whose question string does / does not occur in the development set (lead within that subset).

| reader | benchmark | model | AUROC | needed | estimate | lead | seen strings: share, lead | unseen strings: lead |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | evqa | logistic | 0.73 | 0.881 | +0.04 | -0.40 [-0.86, +0.04] | 21.8%, +0.17 [-0.99, +1.35] | -0.56 [-1.08, -0.05] |
| Qwen3-VL-4B-Instruct | evqa | boosted | 0.718 | 0.881 | +0.03 | -0.36 [-0.69, -0.08] | 21.8%, -0.50 [-1.50, +0.50] | -0.33 [-0.62, -0.09] |
| Qwen3-VL-4B-Instruct | infoseek | logistic | 0.728 | 0.823 | +0.29 | +0.41 [-0.21, +1.01] | 88.6%, +0.43 [-0.25, +1.13] | +0.24 [-1.41, +1.91] |
| Qwen3-VL-4B-Instruct | infoseek | boosted | 0.769 | 0.823 | +0.54 | +1.04 [+0.61, +1.50] | 88.6%, +1.21 [+0.71, +1.73] | -0.24 [-0.77, +0.00] |
| Qwen3-VL-4B-Instruct | okvqa | logistic | 0.624 | 0.857 | +0.00 | -0.35 [-0.72, +0.02] | 5.5%, -0.45 [-2.23, +0.89] | -0.34 [-0.76, +0.05] |
| Qwen3-VL-4B-Instruct | okvqa | boosted | 0.6 | 0.857 | +0.00 | -0.25 [-0.52, +0.00] | 5.5%, +0.00 [-1.34, +1.34] | -0.26 [-0.52, +0.00] |
| Qwen3-VL-4B-Instruct | cvqa | logistic | 0.626 | 0.692 | +0.30 | +0.38 [-0.20, +0.96] | 9.8%, +0.61 [-0.85, +1.93] | +0.36 [-0.25, +0.98] |
| Qwen3-VL-4B-Instruct | cvqa | boosted | 0.599 | 0.692 | +0.12 | +0.02 [-0.52, +0.58] | 9.8%, -0.61 [-2.02, +0.73] | +0.09 [-0.48, +0.67] |
| Qwen2.5-VL-3B-Instruct | evqa | logistic | 0.72 | 0.879 | +0.03 | -0.33 [-0.87, +0.25] | 21.8%, +0.17 [-1.18, +1.64] | -0.47 [-1.09, +0.19] |
| Qwen2.5-VL-3B-Instruct | evqa | boosted | 0.712 | 0.879 | +0.02 | -0.15 [-0.64, +0.34] | 21.8%, -0.17 [-1.64, +1.32] | -0.14 [-0.63, +0.35] |
| Qwen2.5-VL-3B-Instruct | infoseek | logistic | 0.691 | 0.829 | +0.12 | -0.22 [-0.83, +0.36] | 88.6%, +0.00 [-0.56, +0.57] | -1.93 [-4.09, +0.23] |
| Qwen2.5-VL-3B-Instruct | infoseek | boosted | 0.732 | 0.829 | +0.28 | +0.49 [+0.05, +0.97] | 88.6%, +0.68 [+0.21, +1.17] | -0.96 [-2.58, +0.69] |
| Qwen2.5-VL-3B-Instruct | okvqa | logistic | 0.578 | 0.813 | +0.00 | -1.24 [-1.85, -0.67] | 5.5%, -1.79 [-5.36, +1.34] | -1.20 [-1.80, -0.60] |
| Qwen2.5-VL-3B-Instruct | okvqa | boosted | 0.579 | 0.813 | +0.00 | -0.32 [-0.77, +0.12] | 5.5%, -1.34 [-4.46, +1.79] | -0.26 [-0.68, +0.24] |
| InternVL3_5-4B-HF | evqa | logistic | 0.689 | 0.908 | +0.00 | -0.65 [-1.13, -0.18] | 21.8%, -1.67 [-2.98, -0.34] | -0.37 [-0.81, +0.08] |
| InternVL3_5-4B-HF | evqa | boosted | 0.698 | 0.908 | +0.00 | -0.15 [-0.41, +0.08] | 21.8%, -0.17 [-0.84, +0.51] | -0.14 [-0.39, +0.09] |
| InternVL3_5-4B-HF | infoseek | logistic | 0.666 | 0.868 | +0.02 | -0.11 [-0.68, +0.46] | 88.6%, -0.22 [-0.82, +0.40] | +0.72 [+0.00, +1.72] |
| InternVL3_5-4B-HF | infoseek | boosted | 0.716 | 0.868 | +0.10 | +0.33 [-0.11, +0.83] | 88.6%, +0.40 [-0.09, +0.93] | -0.24 [-1.53, +0.98] |
| InternVL3_5-4B-HF | okvqa | logistic | 0.733 | 0.818 | +0.26 | +0.49 [+0.02, +0.99] | 5.5%, -0.89 [-2.68, +0.89] | +0.58 [+0.08, +1.07] |
| InternVL3_5-4B-HF | okvqa | boosted | 0.701 | 0.818 | +0.11 | +0.17 [-0.20, +0.57] | 5.5%, +0.00 [-1.34, +1.34] | +0.18 [-0.21, +0.60] |
| Qwen3-VL-8B-Instruct | evqa | logistic | 0.705 | 0.867 | +0.02 | +0.22 [-0.23, +0.68] | 21.8%, -0.17 [-1.34, +1.01] | +0.33 [-0.14, +0.81] |
| Qwen3-VL-8B-Instruct | evqa | boosted | 0.737 | 0.867 | +0.08 | +0.25 [-0.07, +0.57] | 21.8%, +0.00 [-0.84, +0.99] | +0.33 [+0.04, +0.66] |
| Qwen3-VL-8B-Instruct | infoseek | logistic | 0.621 | 0.868 | +0.00 | -0.74 [-1.18, -0.33] | 88.6%, -0.71 [-1.19, -0.28] | -0.96 [-2.18, +0.00] |
| Qwen3-VL-8B-Instruct | infoseek | boosted | 0.694 | 0.868 | +0.04 | -0.16 [-0.56, +0.19] | 88.6%, -0.25 [-0.66, +0.16] | +0.48 [-0.48, +1.50] |
| Qwen3-VL-8B-Instruct | okvqa | logistic | 0.603 | 0.88 | +0.00 | -0.69 [-1.11, -0.27] | 5.5%, -1.34 [-3.12, +0.00] | -0.65 [-1.10, -0.24] |
| Qwen3-VL-8B-Instruct | okvqa | boosted | 0.587 | 0.88 | +0.00 | -0.25 [-0.52, +0.02] | 5.5%, -0.45 [-1.34, +0.00] | -0.24 [-0.52, +0.05] |

Leads of at least one point with an interval above zero: [('Qwen3-VL-4B-Instruct', 'infoseek', 'boosted')].

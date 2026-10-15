# PREREG v14 — a stronger gate (full hidden states of three layers): forecast

| reader | benchmark | dims | gate | reg | dev-CV AUROC | needed (dev) | forecast G* | held-out AUROC | read % | lead [95%] |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct | evqa | 7691 | lin | 0.001 | 0.662 | 0.857 | +0.00 |  |  |  |
| Qwen3-VL-4B-Instruct | evqa | 7691 | mlp | 0.01 | 0.611 | 0.857 | +0.00 |  |  |  |
| Qwen3-VL-4B-Instruct | infoseek | 7691 | lin | 0.001 | 0.653 | 0.793 | +0.08 |  |  |  |
| Qwen3-VL-4B-Instruct | infoseek | 7691 | mlp | 0.1 | 0.617 | 0.793 | +0.01 |  |  |  |
| Qwen3-VL-4B-Instruct | okvqa | 7691 | lin | 0.001 | 0.612 | 0.869 | +0.00 |  |  |  |
| Qwen3-VL-4B-Instruct | okvqa | 7691 | mlp | 0.01 | 0.589 | 0.869 | +0.00 |  |  |  |
| Qwen2.5-VL-3B-Instruct | evqa | 6155 | lin | 0.001 | 0.632 | 0.824 | +0.00 |  |  |  |
| Qwen2.5-VL-3B-Instruct | evqa | 6155 | mlp | 0.001 | 0.604 | 0.824 | +0.00 |  |  |  |
| Qwen2.5-VL-3B-Instruct | infoseek | 6155 | lin | 0.01 | 0.755 | 0.832 | +0.39 |  |  |  |
| Qwen2.5-VL-3B-Instruct | infoseek | 6155 | mlp | 0.001 | 0.728 | 0.832 | +0.25 |  |  |  |
| Qwen2.5-VL-3B-Instruct | okvqa | 6155 | lin | 0.01 | 0.611 | 0.797 | +0.00 |  |  |  |
| Qwen2.5-VL-3B-Instruct | okvqa | 6155 | mlp | 0.1 | 0.59 | 0.797 | +0.00 |  |  |  |
| InternVL3_5-4B-HF | evqa | 2571 | lin | 0.001 | 0.659 | 0.872 | +0.00 |  |  |  |
| InternVL3_5-4B-HF | evqa | 2571 | mlp | 0.001 | 0.625 | 0.872 | +0.00 |  |  |  |
| InternVL3_5-4B-HF | infoseek | 2571 | lin | 0.01 | 0.681 | 0.848 | +0.06 |  |  |  |
| InternVL3_5-4B-HF | infoseek | 2571 | mlp | 0.01 | 0.631 | 0.848 | +0.01 |  |  |  |
| InternVL3_5-4B-HF | okvqa | 7691 | lin | 0.1 | 0.595 | 0.827 | +0.00 |  |  |  |
| InternVL3_5-4B-HF | okvqa | 7691 | mlp | 0.1 | 0.552 | 0.827 | +0.00 |  |  |  |
| Qwen3-VL-8B-Instruct | evqa | 12299 | lin | 0.001 | 0.68 | 0.889 | +0.00 |  |  |  |
| Qwen3-VL-8B-Instruct | evqa | 12299 | mlp | 0.1 | 0.63 | 0.889 | +0.00 |  |  |  |
| Qwen3-VL-8B-Instruct | infoseek | 12299 | lin | 0.001 | 0.728 | 0.866 | +0.12 |  |  |  |
| Qwen3-VL-8B-Instruct | infoseek | 12299 | mlp | 0.1 | 0.721 | 0.866 | +0.10 |  |  |  |
| Qwen3-VL-8B-Instruct | okvqa | 12299 | lin | 0.001 | 0.595 | 0.851 | +0.00 |  |  |  |
| Qwen3-VL-8B-Instruct | okvqa | 12299 | mlp | 0.1 | 0.593 | 0.851 | +0.00 |  |  |  |
| Qwen3-VL-4B-Instruct | cvqa | 7691 | lin | 0.001 | 0.588 | 0.687 | +0.08 |  |  |  |
| Qwen3-VL-4B-Instruct | cvqa | 7691 | mlp | 0.01 | 0.505 | 0.687 | +0.00 |  |  |  |

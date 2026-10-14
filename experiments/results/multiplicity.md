# Holm correction over the 22 preregistered tests (alpha = 0.05, two-sided, normal approximation to the bootstrap intervals)

Significant after correction: **16 of 22** (unadjusted intervals excluding 0: 17).

| test | prereg | point [95% CI] | p | Holm threshold | significant |
|---|---|---|---|---|---|
| H-SEL (select - always), Qwen3-VL-4B, OK-VQA | v2b | +19.65 [+18.13, +21.23] | 1.8e-135 | 0.0023 | yes |
| H-SEL (select - always), Qwen2.5-VL-3B, OK-VQA | v2b | +13.79 [+12.38, +15.28] | 2.6e-77 | 0.0024 | yes |
| probe, InternVL3.5-4B, E-VQA+InfoSeek | 2026-10-03 | +2.33 [+1.89, +2.76] | 1.9e-25 | 0.0025 | yes |
| probe, Qwen3-VL-8B, E-VQA+InfoSeek | v2c | +1.92 [+1.52, +2.32] | 6.3e-21 | 0.0026 | yes |
| primary, InternVL3.5-4B, E-VQA+InfoSeek | 2026-10-03 | +1.78 [+1.40, +2.17] | 8.1e-20 | 0.0028 | yes |
| primary, Qwen3-VL-4B, E-VQA+InfoSeek | v1 | +1.64 [+1.26, +2.01] | 1.6e-17 | 0.0029 | yes |
| primary, Qwen3-VL-8B, E-VQA+InfoSeek | v2c | +1.63 [+1.23, +1.99] | 4.1e-17 | 0.0031 | yes |
| primary, Qwen2.5-VL-3B, E-VQA+InfoSeek | v1 (replication) | +1.49 [+1.06, +1.89] | 2.8e-12 | 0.0033 | yes |
| probe, Qwen3-VL-8B, OK-VQA | v4c | +1.53 [+1.06, +2.00] | 1.5e-10 | 0.0036 | yes |
| probe, Qwen3-VL-4B, OK-VQA | v2 | +1.55 [+1.06, +2.02] | 2.7e-10 | 0.0038 | yes |
| VGate-full - VGate-pre, Qwen3-VL-4B, OK-VQA | v2b | +0.67 [+0.44, +0.91] | 2e-08 | 0.0042 | yes |
| primary, Qwen3-VL-8B, OK-VQA | v4c | +1.20 [+0.76, +1.63] | 5.1e-08 | 0.0045 | yes |
| VGate-full - VGate-post, Qwen3-VL-4B, OK-VQA | v2b | +1.37 [+0.85, +1.88] | 2e-07 | 0.0050 | yes |
| probe, Qwen2.5-VL-3B, OK-VQA | v2 | +1.21 [+0.66, +1.75] | 1.4e-05 | 0.0056 | yes |
| VGate-full - VGate-pre, Qwen2.5-VL-3B, OK-VQA | v2b | +0.51 [+0.24, +0.79] | 0.00031 | 0.0063 | yes |
| primary, Qwen2.5-VL-3B, OK-VQA | v2 | +0.82 [+0.25, +1.37] | 0.004 | 0.0071 | yes |
| probe, InternVL3.5-4B, OK-VQA | v4c | +0.56 [+0.06, +1.03] | 0.023 | 0.0083 | no |
| primary, InternVL3.5-4B, OK-VQA | v4c | +0.25 [-0.03, +0.51] | 0.064 | 0.0100 | no |
| primary, Qwen3-VL-4B, OK-VQA | v2 | +0.33 [-0.13, +0.76] | 0.15 | 0.0125 | no |
| VGate-full - VGate-post, Qwen2.5-VL-3B, OK-VQA | v2b | +0.20 [-0.24, +0.66] | 0.38 | 0.0167 | no |
| H-OK (gate - never at dev-chosen rate), Qwen2.5-VL-3B, OK-VQA | v2 | -0.02 [-0.27, +0.20] | 0.84 | 0.0250 | no |
| H-OK (gate - never at dev-chosen rate), Qwen3-VL-4B, OK-VQA | v2 | +0.00 [+0.00, +0.00] | 1 | 0.0500 | no |

H-SEL is significant only in the trivial sense recorded in the outcome log (selection recovers never-retrieve).

# Gate learned from one-sided feedback (Qwen3-VL-8B-Instruct, confirm stream, n=10433, 10 orders; PREREG v3b)

Micro accuracy on the pooled stream: never 37.74, always 42.64, better fixed policy per dataset 49.5, oracle 53.75.

| policy | whole stream | last third | read share (last third) |
|---|---|---|---|
| online, shown answer judged only (ε=0.05) | 47.4 (47.1–47.7) | 47.8 (46.9–48.6) | 55.1 (52.6–58.1) |
| online, both outcomes judged | 47.9 (47.8–48.0) | 48.2 (47.5–48.9) | 59.3 (58.2–60.6) |
| VGate-full fitted on dev (frozen) | 48.6 (48.6–48.6) | 48.6 (47.9–49.2) | 59.6 (58.5–60.5) |
| always | 42.6 (42.6–42.6) | 42.9 (41.9–43.7) | 100.0 (100.0–100.0) |
| never | 37.7 (37.7–37.7) | 37.8 (37.2–38.2) | 0.0 (0.0–0.0) |

Mean (min–max) over orders; order variation only.

## Expectations
- O1: held
- O2: held
- O3: held

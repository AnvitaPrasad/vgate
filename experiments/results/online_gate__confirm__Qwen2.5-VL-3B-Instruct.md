# Gate learned from one-sided feedback (Qwen2.5-VL-3B-Instruct, confirm stream, n=10433, 10 orders; PREREG v3b)

Micro accuracy on the pooled stream: never 34.88, always 39.86, better fixed policy per dataset 45.15, oracle 50.11.

| policy | whole stream | last third | read share (last third) |
|---|---|---|---|
| online, shown answer judged only (ε=0.05) | 43.4 (43.0–43.6) | 44.2 (43.3–45.2) | 56.9 (53.6–60.7) |
| online, both outcomes judged | 43.6 (43.4–43.9) | 44.4 (43.8–45.6) | 61.7 (60.9–62.3) |
| VGate-full fitted on dev (frozen) | 44.3 (44.3–44.3) | 44.6 (43.8–45.4) | 58.9 (57.8–59.7) |
| always | 39.9 (39.9–39.9) | 40.2 (39.5–41.2) | 100.0 (100.0–100.0) |
| never | 34.9 (34.9–34.9) | 35.1 (34.6–35.7) | 0.0 (0.0–0.0) |

Mean (min–max) over orders; order variation only.

## Expectations
- O1: held
- O2: held
- O3: held

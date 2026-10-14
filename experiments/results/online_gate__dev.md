# Gate learned from one-sided feedback (dev stream, n=3000, 3 orders; PREREG v3b)

Micro accuracy on the pooled stream: never 34.47, always 39.07, better fixed policy per dataset 46.0, oracle 51.1.

| policy | whole stream | last third | read share (last third) |
|---|---|---|---|
| online, shown answer judged only (ε=0.05) | 43.1 (43.0–43.3) | 44.9 (43.6–45.6) | 56.3 (51.0–60.9) |
| online, both outcomes judged | 43.8 (43.4–44.0) | 44.9 (43.4–45.7) | 59.5 (58.8–59.9) |
| VGate-full fitted on dev (frozen) | 46.2 (46.2–46.2) | 46.9 (46.3–47.5) | 58.5 (58.2–58.7) |
| always | 39.1 (39.1–39.1) | 40.2 (38.6–41.1) | 100.0 (100.0–100.0) |
| never | 34.5 (34.5–34.5) | 34.3 (33.6–35.3) | 0.0 (0.0–0.0) |

Mean (min–max) over orders; order variation only.

## Expectations
- O1: held
- O2: held
- O3: NOT held

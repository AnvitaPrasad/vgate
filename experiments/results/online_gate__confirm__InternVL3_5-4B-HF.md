# Gate learned from one-sided feedback (InternVL3_5-4B-HF, confirm stream, n=10433, 10 orders; PREREG v3b)

Micro accuracy on the pooled stream: never 25.6, always 31.16, better fixed policy per dataset 38.02, oracle 42.69.

| policy | whole stream | last third | read share (last third) |
|---|---|---|---|
| online, shown answer judged only (ε=0.05) | 36.2 (35.9–36.6) | 37.1 (36.2–38.1) | 62.1 (60.7–64.9) |
| online, both outcomes judged | 36.5 (36.4–36.8) | 37.2 (36.5–38.1) | 61.7 (60.8–62.3) |
| VGate-full fitted on dev (frozen) | 37.1 (37.1–37.1) | 37.2 (36.7–38.0) | 60.0 (58.9–60.6) |
| always | 31.2 (31.2–31.2) | 31.4 (30.6–32.6) | 100.0 (100.0–100.0) |
| never | 25.6 (25.6–25.6) | 25.8 (25.1–26.4) | 0.0 (0.0–0.0) |

Mean (min–max) over orders; order variation only.

## Expectations
- O1: held
- O2: held
- O3: held

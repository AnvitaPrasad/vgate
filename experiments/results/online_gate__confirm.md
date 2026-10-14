# Gate learned from one-sided feedback (Qwen3-VL-4B-Instruct, confirm stream, n=10433, 10 orders; PREREG v3b)

Micro accuracy on the pooled stream: never 35.38, always 38.52, better fixed policy per dataset 46.14, oracle 50.89.

| policy | whole stream | last third | read share (last third) |
|---|---|---|---|
| online, shown answer judged only (ε=0.05) | 44.3 (44.1–44.5) | 45.0 (44.0–45.7) | 55.8 (54.6–56.9) |
| online, both outcomes judged | 44.8 (44.6–45.0) | 45.2 (44.2–46.3) | 58.4 (57.7–59.2) |
| VGate-full fitted on dev (frozen) | 45.3 (45.3–45.3) | 45.3 (44.4–46.3) | 55.2 (54.4–56.0) |
| always | 38.5 (38.5–38.5) | 38.7 (37.6–39.3) | 100.0 (100.0–100.0) |
| never | 35.4 (35.4–35.4) | 35.3 (34.9–36.1) | 0.0 (0.0–0.0) |

Mean (min–max) over orders; order variation only.

## Expectations
- O1: held
- O2: held
- O3: held

Exploratory, live-retriever logs (n=800, micro accuracy, cold start): never 39.38, always 25.62, dev-fitted gate transferred 36.62, online one-sided 37.0 (35.2–38.2) (second half of the stream 38.5 (35.0–40.5)).
